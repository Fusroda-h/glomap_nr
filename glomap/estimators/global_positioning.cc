#include "glomap/estimators/global_positioning.h"
#include "glomap/estimators/cost_function.h"

#include <numeric>
#include <queue>
#include <unordered_set>
#include <fstream>
#include <algorithm>  

namespace glomap {
namespace {

// Return observation index (feature id) of image_id inside a track whose
// observations are stored as std::vector<std::pair<image_t, feature_id>>.
// Returns -1 if not found.
int FindObs(const Track& track, const image_t image_id) {
  for (const auto& obs : track.observations) {
    if (obs.first == image_id) {
      return static_cast<int>(obs.second);
    }
  }
  return -1;
}

// Normalize vector safely.
Eigen::Vector3d NormalizeSafe(const Eigen::Vector3d& v, const double eps = 1e-12) {
  const double n = v.norm();
  if (n < eps) return Eigen::Vector3d(0, 0, 0);
  return v / n;
}

// Build edge key (u,v) -> uint64
uint64_t EdgeKey(const image_t u, const image_t v) {
  const uint64_t a = static_cast<uint64_t>(std::min(u, v));
  const uint64_t b = static_cast<uint64_t>(std::max(u, v));
  return (a << 32) | b;
}

// Random 3D vector in box [low, high]
Eigen::Vector3d RandVector3d(std::mt19937& random_generator,
                             double low,
                             double high) {
  std::uniform_real_distribution<double> distribution(low, high);
  return Eigen::Vector3d(distribution(random_generator),
                         distribution(random_generator),
                         distribution(random_generator));
}

// Read relative pose i -> j from view_graph by scanning image_pairs.
// Returns Rji, t_hat_ji such that X_j = Rji * X_i + s * t_hat_ji.
bool GetRelPose_I_to_J(const ViewGraph& view_graph,
                       const image_t i,
                       const image_t j,
                       Eigen::Matrix3d* Rji,
                       Eigen::Vector3d* tji_hat) {
  for (const auto& kv : view_graph.image_pairs) {
    const auto& ip = kv.second;
    if (!ip.is_valid) continue;
    const image_t id1 = ip.image_id1;
    const image_t id2 = ip.image_id2;

    // stored as cam2_from_cam1: X_2 = R_21 * X_1 + s * t_21
    const Eigen::Matrix3d R21 = ip.cam2_from_cam1.rotation.toRotationMatrix();
    const Eigen::Vector3d t21 = ip.cam2_from_cam1.translation;

    if (id1 == i && id2 == j) {
      // wanted i -> j, and we have i -> j
      *Rji = R21;
      *tji_hat = NormalizeSafe(t21);
      return true;
    } else if (id1 == j && id2 == i) {
      // wanted i -> j, but we have j -> i, so invert
      const Eigen::Matrix3d R12 = R21.transpose();
      const Eigen::Vector3d t12 = -(R21.transpose() * t21);
      *Rji = R12;
      *tji_hat = NormalizeSafe(t12);
      return true;
    }
  }
  return false;
}

// Canonicalized triplet key (i < j < k)
struct TripletKey {
  image_t i, j, k;
  bool operator==(const TripletKey& other) const {
    return i == other.i && j == other.j && k == other.k;
  }
};

struct TripletKeyHash {
  std::size_t operator()(const TripletKey& t) const {
    // very simple hash
    return (static_cast<std::size_t>(t.i) * 1315423911u) ^
           (static_cast<std::size_t>(t.j) << 16) ^
           (static_cast<std::size_t>(t.k) << 1);
  }
};

}  // namespace

// ------------------------------------------------------------------
// GlobalPositioner
// ------------------------------------------------------------------

GlobalPositioner::GlobalPositioner(const GlobalPositionerOptions& options)
    : options_(options) {
  random_generator_.seed(options_.seed);
}

bool GlobalPositioner::Solve(const ViewGraph& view_graph,
                             std::unordered_map<camera_t, Camera>& cameras,
                             std::unordered_map<image_t, Image>& images,
                             std::unordered_map<track_t, Track>& tracks) {
  if (images.empty()) {
    LOG(ERROR) << "Number of images = " << images.size();
    return false;
  }
  if (view_graph.image_pairs.empty() &&
      options_.constraint_type != GlobalPositionerOptions::ONLY_POINTS) {
    LOG(ERROR) << "Number of image_pairs = " << view_graph.image_pairs.size();
    return false;
  }
  if (tracks.empty() &&
      options_.constraint_type != GlobalPositionerOptions::ONLY_CAMERAS) {
    LOG(ERROR) << "Number of tracks = " << tracks.size();
    return false;
  }

  LOG(INFO) << "[GlobalPositioner] center_init_mode enum = "
          << static_cast<int>(options_.center_init_mode);

  // If ONLY_POINTS: initialize cameras using triplet RANSAC from directions.
  switch (options_.center_init_mode) {
    case GlobalPositionerOptions::CenterInitMode::RANDOM: {
      // fallback to the original random initialization
      LOG(INFO) << "[GlobalPositioner] CenterInitMode = RANDOM (0)";
      InitializeRandomPositions(view_graph, images, tracks);
      
      break;
      }
    case GlobalPositionerOptions::CenterInitMode::SCALED_TRIPLET: {
      LOG(INFO) << "[GlobalPositioner] CenterInitMode = SCALED_TRIPLET (1)";
      std::unordered_map<uint64_t, std::vector<EdgeScaleSample>> edge_scales;
      EstimateEdgeScalesByTriRansac(view_graph,
                                    images,
                                    tracks,
                                    cameras,
                                    edge_scales);
      InitializeCamerasFromTriScales(view_graph, images, edge_scales);
      InitializePointsFromCameras(cameras, images, tracks);
      break;
      }
  }

  if (options_.dump_init_centers_csv) {
    DumpInitialCentersCSV(images, options_.init_centers_csv_path);
  }

  // Setup ceres problem
  LOG(INFO) << "Setting up the global positioner problem";
  SetupProblem(view_graph, tracks);

  // Add camera-to-camera constraints unless ONLY_POINTS.
  if (options_.constraint_type != GlobalPositionerOptions::ONLY_POINTS) {
    AddCameraToCameraConstraints(view_graph, images);
  }

  // Add point-to-camera constraints unless ONLY_CAMERAS.
  if (options_.constraint_type != GlobalPositionerOptions::ONLY_CAMERAS) {
    AddPointToCameraConstraints(cameras, images, tracks);
  }

  AddCamerasAndPointsToParameterGroups(images, tracks);
  ParameterizeVariables(images, tracks);

  LOG(INFO) << "Solving the global positioner problem";

  ceres::Solver::Summary summary;
  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solve(options_.solver_options, problem_.get(), &summary);

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
  }

  ConvertResults(images);
  return summary.IsSolutionUsable();
}

void GlobalPositioner::SetupProblem(
    const ViewGraph& view_graph,
    const std::unordered_map<track_t, Track>& tracks) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);

  scales_.clear();
  scales_.reserve(
      view_graph.image_pairs.size() +
      std::accumulate(tracks.begin(),
                      tracks.end(),
                      0,
                      [](int sum, const std::pair<track_t, Track>& track) {
                        return sum + track.second.observations.size();
                      }));
}

void GlobalPositioner::InitializeRandomPositions(
    const ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  std::unordered_set<image_t> constrained_positions;
  constrained_positions.reserve(images.size());
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;
    constrained_positions.insert(image_pair.image_id1);
    constrained_positions.insert(image_pair.image_id2);
  }

  if (options_.constraint_type != GlobalPositionerOptions::ONLY_CAMERAS) {
    for (const auto& [track_id, track] : tracks) {
      if (track.observations.size() < options_.min_num_view_per_track) continue;
      for (const auto& observation : track.observations) {
        const image_t img_id = observation.first;
        auto it_img = images.find(img_id);
        if (it_img == images.end()) continue;
        if (!it_img->second.is_registered) continue;
        constrained_positions.insert(img_id);
      }
    }
  }

  if (!options_.generate_random_positions || !options_.optimize_positions) {
    for (auto& [image_id, image] : images) {
      image.cam_from_world.translation = image.Center();
    }
    return;
  }

  for (auto& [image_id, image] : images) {
    if (constrained_positions.count(image_id)) {
      image.cam_from_world.translation =
          100.0 * RandVector3d(random_generator_, -1, 1);
    } else {
      image.cam_from_world.translation = image.Center();
    }
  }

  VLOG(2) << "Constrained positions: " << constrained_positions.size();
}

void GlobalPositioner::AddCameraToCameraConstraints(
    const ViewGraph& view_graph, std::unordered_map<image_t, Image>& images) {
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;

    const image_t image_id1 = image_pair.image_id1;
    const image_t image_id2 = image_pair.image_id2;
    if (images.find(image_id1) == images.end() ||
        images.find(image_id2) == images.end()) {
      continue;
    }

    CHECK_GT(scales_.capacity(), scales_.size())
        << "Not enough capacity was reserved for the scales.";
    double& scale = scales_.emplace_back(1);

    const Eigen::Vector3d translation =
        -(images[image_id2].cam_from_world.rotation.inverse().toRotationMatrix() *
          image_pair.cam2_from_cam1.translation);
    ceres::CostFunction* cost_function =
        BATAPairwiseDirectionError::Create(translation);
    problem_->AddResidualBlock(
        cost_function,
        options_.loss_function.get(),
        images[image_id1].cam_from_world.translation.data(),
        images[image_id2].cam_from_world.translation.data(),
        &scale);

    problem_->SetParameterLowerBound(&scale, 0, 1e-5);
  }

  VLOG(2) << problem_->NumResidualBlocks()
          << " camera to camera constraints were added to the position "
             "estimation problem.";
}

void GlobalPositioner::AddPointToCameraConstraints(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  const size_t num_cam_to_cam = problem_->NumResidualBlocks();
  const size_t num_pt_to_cam = tracks.size();

  VLOG(2) << num_pt_to_cam
          << " point to camera constriants were added to the position "
             "estimation problem.";

  if (num_pt_to_cam == 0) return;

  double weight_scale_pt = 1.0;
  if (num_cam_to_cam > 0 &&
      options_.constraint_type ==
          GlobalPositionerOptions::POINTS_AND_CAMERAS_BALANCED) {
    weight_scale_pt = options_.constraint_reweight_scale *
                      static_cast<double>(num_cam_to_cam) /
                      static_cast<double>(num_pt_to_cam);
  }
  VLOG(2) << "Point to camera weight scaled: " << weight_scale_pt;

  if (loss_function_ptcam_uncalibrated_ == nullptr) {
    loss_function_ptcam_uncalibrated_ =
        std::make_shared<ceres::ScaledLoss>(options_.loss_function.get(),
                                            0.5 * weight_scale_pt,
                                            ceres::DO_NOT_TAKE_OWNERSHIP);
  }

  if (options_.constraint_type ==
      GlobalPositionerOptions::POINTS_AND_CAMERAS_BALANCED) {
    loss_function_ptcam_calibrated_ =
        std::make_shared<ceres::ScaledLoss>(options_.loss_function.get(),
                                            weight_scale_pt,
                                            ceres::DO_NOT_TAKE_OWNERSHIP);
  } else {
    loss_function_ptcam_calibrated_ = options_.loss_function;
  }

  for (auto& [track_id, track] : tracks) {
    if (track.observations.size() < options_.min_num_view_per_track) continue;

    if (options_.optimize_points && options_.generate_random_points) {
      track.xyz = 100.0 * RandVector3d(random_generator_, -1, 1);
      track.is_initialized = true;
    }

    AddTrackToProblem(track_id, cameras, images, tracks);
  }
}

void GlobalPositioner::AddTrackToProblem(
    track_t track_id,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  for (const auto& observation : tracks[track_id].observations) {
    const image_t img_id = observation.first;
    if (images.find(img_id) == images.end()) continue;

    Image& image = images[img_id];
    if (!image.is_registered) continue;

    const Eigen::Vector3d& feature_undist =
        image.features_undist[observation.second];
    if (feature_undist.array().isNaN().any()) {
      LOG(WARNING)
          << "Ignoring feature because it failed to undistort: track_id="
          << track_id << ", image_id=" << img_id
          << ", feature_id=" << observation.second;
      continue;
    }

    const Eigen::Vector3d translation =
        image.cam_from_world.rotation.inverse().toRotationMatrix() *
        image.features_undist[observation.second];
    ceres::CostFunction* cost_function =
        BATAPairwiseDirectionError::Create(translation);

    CHECK_GT(scales_.capacity(), scales_.size())
        << "Not enough capacity was reserved for the scales.";
    double& scale = scales_.emplace_back(1);
    if (!options_.generate_scales && tracks[track_id].is_initialized) {
      const Eigen::Vector3d trans_calc =
          tracks[track_id].xyz - image.cam_from_world.translation;
      scale = std::max(1e-5,
                       translation.dot(trans_calc) / trans_calc.squaredNorm());
    }

    if (cameras[image.camera_id].has_prior_focal_length) {
      problem_->AddResidualBlock(cost_function,
                                 loss_function_ptcam_calibrated_.get(),
                                 image.cam_from_world.translation.data(),
                                 tracks[track_id].xyz.data(),
                                 &scale);
    } else {
      problem_->AddResidualBlock(cost_function,
                                 loss_function_ptcam_uncalibrated_.get(),
                                 image.cam_from_world.translation.data(),
                                 tracks[track_id].xyz.data(),
                                 &scale);
    }

    problem_->SetParameterLowerBound(&scale, 0, 1e-5);
  }
}

void GlobalPositioner::AddCamerasAndPointsToParameterGroups(
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();

  // group 0: edge scales
  for (double& scale : scales_) {
    parameter_ordering->AddElementToGroup(&scale, 0);
  }

  int group_id = 1;
  if (!tracks.empty()) {
    for (auto& [track_id, track] : tracks) {
      if (problem_->HasParameterBlock(track.xyz.data()))
        parameter_ordering->AddElementToGroup(track.xyz.data(), group_id);
    }
    group_id++;
  }

  for (auto& [image_id, image] : images) {
    if (problem_->HasParameterBlock(image.cam_from_world.translation.data())) {
      parameter_ordering->AddElementToGroup(
          image.cam_from_world.translation.data(), group_id);
    }
  }
}

void GlobalPositioner::ParameterizeVariables(
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  if (!options_.optimize_positions) {
    for (auto& [image_id, image] : images)
      if (problem_->HasParameterBlock(image.cam_from_world.translation.data()))
        problem_->SetParameterBlockConstant(
            image.cam_from_world.translation.data());
  }

  if (!options_.optimize_points) {
    for (auto& [track_id, track] : tracks) {
      if (problem_->HasParameterBlock(track.xyz.data())) {
        problem_->SetParameterBlockConstant(track.xyz.data());
      }
    }
  }

  if (!options_.optimize_scales) {
    for (double& scale : scales_) {
      problem_->SetParameterBlockConstant(&scale);
    }
  }

  if (!tracks.empty()) {
    options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    options_.solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
  } else {
    options_.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options_.solver_options.preconditioner_type = ceres::JACOBI;
  }
}

void GlobalPositioner::ConvertResults(
    std::unordered_map<image_t, Image>& images) {
  for (auto& [image_id, image] : images) {
    image.cam_from_world.translation =
        -(image.cam_from_world.rotation.toRotationMatrix() *
          image.cam_from_world.translation);
  }
}

void GlobalPositioner::DumpInitialCentersCSV(
    const std::unordered_map<image_t, Image>& images,
    const std::string& csv_path) const {
  std::ofstream csv(csv_path, std::ios::out);
  if (!csv.is_open()) {
    LOG(ERROR) << "Failed to open CSV file for writing: " << csv_path;
    return;
  }

  csv << "image_id,cx,cy,cz\n";
  for (const auto& [img_id, img] : images) {
    const Eigen::Vector3d C = img.cam_from_world.translation;
    csv << img_id << "," << C.x() << "," << C.y() << "," << C.z() << "\n";
  }
}

// ------------------------------------------------------------------
// NEW: triplet-based parts
// ------------------------------------------------------------------
bool GlobalPositioner::EstimateSForTripletRansac(
    image_t i, image_t j, image_t k,
    const std::vector<track_t>& tids,
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<track_t, Track>& tracks,
    const std::unordered_map<camera_t, Camera>& cameras,
    Eigen::Vector3d* s_out,
    int* best_inliers_out,
    double* median_err_out,
    std::vector<Eigen::Matrix3d>* A_list_out) {

  if (A_list_out) {
    A_list_out->clear();
  }

  // 1) get relative poses i->j, j->k, i->k
  Eigen::Matrix3d Rij, Rjk, Rik;
  Eigen::Vector3d t_hat_ij, t_hat_jk, t_hat_ik;
  if (!GetRelPose_I_to_J(view_graph, i, j, &Rij, &t_hat_ij)) return false;
  if (!GetRelPose_I_to_J(view_graph, j, k, &Rjk, &t_hat_jk)) return false;
  if (!GetRelPose_I_to_J(view_graph, i, k, &Rik, &t_hat_ik)) return false;

  // 2) intrinsics per view (we need them for reprojection scoring)
  double fix, fiy, cix, ciy;
  {
    const auto& cam = cameras.at(images.at(i).camera_id);
    fix = cam.params[0]; fiy = cam.params[1];
    cix = cam.params[2]; ciy = cam.params[3];
  }
  double fjx, fjy, cjx, cjy;
  {
    const auto& cam = cameras.at(images.at(j).camera_id);
    fjx = cam.params[0]; fjy = cam.params[1];
    cjx = cam.params[2]; cjy = cam.params[3];
  }
  double fkx, fky, ckx, cky;
  {
    const auto& cam = cameras.at(images.at(k).camera_id);
    fkx = cam.params[0]; fky = cam.params[1];
    ckx = cam.params[2]; cky = cam.params[3];
  }

  const int max_iters = options_.tri_ransac_max_iters;
  const double px_thr = options_.tri_inlier_px_thresh;
  const double min_depth = options_.tri_min_depth;

  int best_inl = -1;
  Eigen::Vector3d best_s = Eigen::Vector3d::Zero();
  std::vector<double> best_errs;

  // RANSAC: sample 1 triplet-point, solve s, score on all 3-view tracks
  for (int it = 0; it < max_iters; ++it) {
    if (tids.empty()) break;

    const track_t t0 = tids[it % tids.size()];

    const int fi = FindObs(tracks.at(t0), i);
    const int fj = FindObs(tracks.at(t0), j);
    const int fk = FindObs(tracks.at(t0), k);
    if (fi < 0 || fj < 0 || fk < 0) {
      continue;
    }

    // direction in cam-i for the minimal sample
    const Eigen::Vector3d di = NormalizeSafe(images.at(i).features_undist[fi]);

    // ---- minimal solve for s = (s_ij, s_jk, s_ik) ----
    // (Rjk*Rij - Rik) * di = s_ij * (Rjk*t_ij) + s_jk * t_jk - s_ik * t_ik
    const Eigen::Vector3d B = (Rjk * Rij - Rik) * di;
    Eigen::Matrix3d A;
    A.col(0) = B.cross(Rjk * t_hat_ij);
    A.col(1) = B.cross(t_hat_jk);
    A.col(2) = B.cross(-t_hat_ik);

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d s = svd.matrixV().col(2);
    if (!s.allFinite() || s.norm() < 1e-12) {
      continue;
    }
    // s.normalize();  // scale is up-to-scale anyway

    // ---- score this hypothesis on all tracks that see i,j,k ----
    int inl = 0;
    std::vector<double> cur_errs;
    cur_errs.reserve(tids.size());

    for (const track_t tid : tids) {
      const auto& tr = tracks.at(tid);
      const int fi2 = FindObs(tr, i);
      const int fj2 = FindObs(tr, j);
      const int fk2 = FindObs(tr, k);
      if (fi2 < 0 || fj2 < 0 || fk2 < 0) continue;

      const Eigen::Vector3d di2 = NormalizeSafe(images.at(i).features_undist[fi2]);
      const Eigen::Vector3d dj2 = NormalizeSafe(images.at(j).features_undist[fj2]);
      const Eigen::Vector3d dk2 = NormalizeSafe(images.at(k).features_undist[fk2]);

      const Eigen::Vector3d tij = s(0) * t_hat_ij;
      const Eigen::Vector3d tjk = s(1) * t_hat_jk;
      const Eigen::Vector3d tik = s(2) * t_hat_ik;

      // 9x3 system to solve depths (lambda_i, lambda_j, lambda_k)
      Eigen::Matrix<double, 9, 3> A9;
      Eigen::Matrix<double, 9, 1> b9;
      A9.setZero(); b9.setZero();

      // i -> j
      A9.block<3,1>(0,0) = -(Rij * di2);
      A9.block<3,1>(0,1) =  dj2;
      b9.segment<3>(0)   =  tij;

      // j -> k
      A9.block<3,1>(3,1) = -(Rjk * dj2);
      A9.block<3,1>(3,2) =  dk2;
      b9.segment<3>(3)   =  tjk;

      // i -> k
      A9.block<3,1>(6,0) = -(Rik * di2);
      A9.block<3,1>(6,2) =  dk2;
      b9.segment<3>(6)   =  tik;

      Eigen::Vector3d lambda = A9.colPivHouseholderQr().solve(b9);
      if (!lambda.allFinite()) continue;
      if (lambda(0) <= min_depth ||
          lambda(1) <= min_depth ||
          lambda(2) <= min_depth) {
        continue;
      }

      const Eigen::Vector3d Xi = lambda(0) * di2;       // cam-i
      const Eigen::Vector3d Xj = Rij * Xi + tij;        // cam-j
      const Eigen::Vector3d Xk = Rjk * Xj + tjk;        // cam-k
      if (Xi.z() <= min_depth || Xj.z() <= min_depth || Xk.z() <= min_depth) {
        continue;
      }

      auto proj = [](const Eigen::Vector3d& X,
                     double fx, double fy, double cx, double cy) {
        const double z = std::max(X.z(), 1e-9);
        return Eigen::Vector2d(fx * (X.x() / z) + cx,
                               fy * (X.y() / z) + cy);
      };
      const Eigen::Vector2d pi = proj(Xi, fix, fiy, cix, ciy);
      const Eigen::Vector2d pj = proj(Xj, fjx, fjy, cjx, cjy);
      const Eigen::Vector2d pk = proj(Xk, fkx, fky, ckx, cky);

      const Eigen::Vector2d ui = proj(di2, fix, fiy, cix, ciy);
      const Eigen::Vector2d uj = proj(dj2, fjx, fjy, cjx, cjy);
      const Eigen::Vector2d uk = proj(dk2, fkx, fky, ckx, cky);

      const double e = (pi - ui).norm() + (pj - uj).norm() + (pk - uk).norm();

      if (e < 3.0 * px_thr) {
        ++inl;
        cur_errs.push_back(e / 3.0);  // avg per view
      }
    }

    if (inl > best_inl) {
      best_inl = inl;
      best_s = s;
      best_errs = std::move(cur_errs);
    }
  }  // RANSAC loop

  // RANSAC gating
  if (best_inl < options_.tri_min_inliers) {
    return false;
  }

  double median_err = 0.0;
  if (!best_errs.empty()) {
    std::nth_element(best_errs.begin(),
                     best_errs.begin() + best_errs.size() / 2,
                     best_errs.end());
    median_err = best_errs[best_errs.size() / 2];
  }
  const double kReprojFactor = 1.2;
  if (median_err > kReprojFactor * px_thr) {
    return false;
  }

  // Export best_s and stats
  if (s_out) *s_out = best_s;
  if (best_inliers_out) *best_inliers_out = best_inl;
  if (median_err_out) *median_err_out = median_err;

  // ------------------------------------------------------------------
  // Build A_local list for all inlier tracks under the best hypothesis.
  // This will be used later in the global LS SVD.
  // ------------------------------------------------------------------
  if (!A_list_out) {
    return true;
  }

  A_list_out->clear();

  const Eigen::Vector3d tij_best = best_s(0) * t_hat_ij;
  const Eigen::Vector3d tjk_best = best_s(1) * t_hat_jk;
  const Eigen::Vector3d tik_best = best_s(2) * t_hat_ik;

  for (const track_t tid : tids) {
    const auto& tr = tracks.at(tid);
    const int fi2 = FindObs(tr, i);
    const int fj2 = FindObs(tr, j);
    const int fk2 = FindObs(tr, k);
    if (fi2 < 0 || fj2 < 0 || fk2 < 0) continue;

    const Eigen::Vector3d di2 = NormalizeSafe(images.at(i).features_undist[fi2]);
    const Eigen::Vector3d dj2 = NormalizeSafe(images.at(j).features_undist[fj2]);
    const Eigen::Vector3d dk2 = NormalizeSafe(images.at(k).features_undist[fk2]);

    // Depth solve again under best hypothesis to check inlier condition.
    Eigen::Matrix<double, 9, 3> A9;
    Eigen::Matrix<double, 9, 1> b9;
    A9.setZero(); b9.setZero();

    A9.block<3,1>(0,0) = -(Rij * di2);
    A9.block<3,1>(0,1) =  dj2;
    b9.segment<3>(0)   =  tij_best;

    A9.block<3,1>(3,1) = -(Rjk * dj2);
    A9.block<3,1>(3,2) =  dk2;
    b9.segment<3>(3)   =  tjk_best;

    A9.block<3,1>(6,0) = -(Rik * di2);
    A9.block<3,1>(6,2) =  dk2;
    b9.segment<3>(6)   =  tik_best;

    Eigen::Vector3d lambda = A9.colPivHouseholderQr().solve(b9);
    if (!lambda.allFinite()) continue;
    if (lambda(0) <= min_depth ||
        lambda(1) <= min_depth ||
        lambda(2) <= min_depth) {
      continue;
    }

    const Eigen::Vector3d Xi = lambda(0) * di2;
    const Eigen::Vector3d Xj = Rij * Xi + tij_best;
    const Eigen::Vector3d Xk = Rjk * Xj + tjk_best;
    if (Xi.z() <= min_depth || Xj.z() <= min_depth || Xk.z() <= min_depth) {
      continue;
    }

    auto proj = [](const Eigen::Vector3d& X,
                   double fx, double fy, double cx, double cy) {
      const double z = std::max(X.z(), 1e-9);
      return Eigen::Vector2d(fx * (X.x() / z) + cx,
                             fy * (X.y() / z) + cy);
    };
    const Eigen::Vector2d pi = proj(Xi, fix, fiy, cix, ciy);
    const Eigen::Vector2d pj = proj(Xj, fjx, fjy, cjx, cjy);
    const Eigen::Vector2d pk = proj(Xk, fkx, fky, ckx, cky);

    const Eigen::Vector2d ui = proj(di2, fix, fiy, cix, ciy);
    const Eigen::Vector2d uj = proj(dj2, fjx, fjy, cjx, cjy);
    const Eigen::Vector2d uk = proj(dk2, fkx, fky, ckx, cky);

    const double e = (pi - ui).norm() + (pj - uj).norm() + (pk - uk).norm();
    if (e >= 3.0 * px_thr) {
      continue;
    }

    // Build local 3x3 constraint: A_local * [s_ij, s_jk, s_ik]^T ≈ 0
    const Eigen::Vector3d B = (Rjk * Rij - Rik) * di2;
    Eigen::Matrix3d A_local;
    A_local.col(0) = B.cross(Rjk * t_hat_ij);
    A_local.col(1) = B.cross(t_hat_jk);
    A_local.col(2) = B.cross(-t_hat_ik);

    A_list_out->push_back(A_local);
  }

  return true;
}

void GlobalPositioner::EstimateEdgeScalesByTriRansac(
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<track_t, Track>& tracks,
    const std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<uint64_t, std::vector<EdgeScaleSample>>& edge_scales) {
  edge_scales.clear();

  // 1) Build local triplet -> tids map in O(#tracks * (#obs_in_track)^3)
  using TripletMap =
      std::unordered_map<TripletKey, std::vector<track_t>, TripletKeyHash>;
  TripletMap triplet_map;
  triplet_map.reserve(tracks.size());

  for (const auto& [tid, tr] : tracks) {
    const auto& obs = tr.observations;
    if (obs.size() < 3) continue;

    std::vector<image_t> img_ids;
    img_ids.reserve(obs.size());
    for (const auto& o : obs) {
      img_ids.push_back(o.first);
    }

    const int n = static_cast<int>(img_ids.size());
    int emitted = 0;
    for (int a = 0; a < n; ++a) {
      for (int b = a + 1; b < n; ++b) {
        for (int c = b + 1; c < n; ++c) {
          if (emitted >= options_.tri_max_triplets_per_track) break;

          image_t ia = img_ids[a];
          image_t ib = img_ids[b];
          image_t ic = img_ids[c];
          if (ib < ia) std::swap(ia, ib);
          if (ic < ib) std::swap(ib, ic);
          if (ib < ia) std::swap(ia, ib);

          TripletKey key{ia, ib, ic};
          triplet_map[key].push_back(tid);
          ++emitted;
        }
      }
    }
  }

  // Container for all local constraints: each corresponds to one inlier track of a valid triplet.
  struct TripletConstraint {
    image_t i, j, k;
    Eigen::Matrix3d A_local;
  };
  std::vector<TripletConstraint> constraints;
  constraints.reserve(triplet_map.size() * 4);  // rough guess

  int num_triplets_total   = 0;
  int num_triplets_success = 0;

  for (const auto& kv : triplet_map) {
    const TripletKey& key = kv.first;
    const std::vector<track_t>& tids3 = kv.second;
    ++num_triplets_total;

    // Require a minimum number of supporting 3-view tracks.
    if (static_cast<int>(tids3.size()) < options_.tri_min_inliers) {
      continue;
    }

    Eigen::Vector3d s_ijk;
    int best_inl = 0;
    double median_err = 0.0;
    std::vector<Eigen::Matrix3d> A_list;

    if (EstimateSForTripletRansac(key.i, key.j, key.k,
                                  tids3,
                                  view_graph, images, tracks, cameras,
                                  &s_ijk,
                                  &best_inl,
                                  &median_err,
                                  &A_list)) {
      if (!A_list.empty()) {
        ++num_triplets_success;
        for (const auto& A_local : A_list) {
          TripletConstraint tc;
          tc.i = key.i;
          tc.j = key.j;
          tc.k = key.k;
          tc.A_local = A_local;
          constraints.push_back(std::move(tc));
        }
      }
    }
  }

  const double kMinEdgeScale = 1e-4;

  LOG(INFO) << "[TRI_RANSAC] triplets (unique) = " << num_triplets_total
            << ", success = " << num_triplets_success
            << ", success ratio = "
            << (num_triplets_total > 0
                    ? static_cast<double>(num_triplets_success)
                          / static_cast<double>(num_triplets_total)
                    : 0.0)
            << ", local constraints (3x3 blocks) = " << constraints.size();

  // If no constraints survived, nothing to do.
  if (constraints.empty()) {
    LOG(WARNING) << "[GLOBAL_LS] No valid triplet constraints; edge scales remain empty.";
    return;
  }

  // 2) Build global linear system: Aglob * s ≈ 0
  //    - each (i,j,k, A_local) adds 3 rows and touches 3 edge variables.
  struct GlobalConstraint {
    int e_ij;
    int e_jk;
    int e_ik;
    Eigen::Matrix3d A_local;
  };

  std::unordered_map<uint64_t, int> edge_to_id;
  edge_to_id.reserve(constraints.size() * 2);

  auto get_edge_id = [&edge_to_id](uint64_t key) -> int {
    auto it = edge_to_id.find(key);
    if (it != edge_to_id.end()) {
      return it->second;
    }
    const int new_id = static_cast<int>(edge_to_id.size());
    edge_to_id.emplace(key, new_id);
    return new_id;
  };

  std::vector<GlobalConstraint> gconstraints;
  gconstraints.reserve(constraints.size());

  for (const auto& tc : constraints) {
    const uint64_t key_ij = EdgeKey(tc.i, tc.j);
    const uint64_t key_jk = EdgeKey(tc.j, tc.k);
    const uint64_t key_ik = EdgeKey(tc.i, tc.k);

    GlobalConstraint gc;
    gc.e_ij = get_edge_id(key_ij);
    gc.e_jk = get_edge_id(key_jk);
    gc.e_ik = get_edge_id(key_ik);
    gc.A_local = tc.A_local;
    gconstraints.push_back(std::move(gc));
  }

  const int num_edges = static_cast<int>(edge_to_id.size());
  const int num_rows  = 3 * static_cast<int>(gconstraints.size());

  LOG(INFO) << "[GLOBAL_LS] A size: " << num_rows << " x " << num_edges
            << " (rows = 3 * #constraints, cols = #edges)";

  if (num_edges == 0) {
    LOG(WARNING) << "[GLOBAL_LS] No edges created from triplets.";
    return;
  }

  Eigen::MatrixXd Aglob(num_rows, num_edges);
  Aglob.setZero();

  for (int idx = 0; idx < static_cast<int>(gconstraints.size()); ++idx) {
    const int r = 3 * idx;
    const auto& gc = gconstraints[idx];
    // Note: A_local is 3x3; each column corresponds to one of (s_ij, s_jk, s_ik).
    Aglob.block<3,1>(r, gc.e_ij) = gc.A_local.col(0);
    Aglob.block<3,1>(r, gc.e_jk) = gc.A_local.col(1);
    Aglob.block<3,1>(r, gc.e_ik) = gc.A_local.col(2);
  }

  // 3) Solve Aglob * s ≈ 0 using SVD: smallest singular value gives the nullspace.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      Aglob, Eigen::ComputeThinV);
  const Eigen::VectorXd singular_values = svd.singularValues();
  Eigen::VectorXd s_global = svd.matrixV().col(num_edges - 1);

  const int topk = std::min(10, static_cast<int>(singular_values.size()));
  LOG(INFO) << "[GLOBAL_LS] singular values (top " << topk << "):";
  for (int i = 0; i < topk; ++i) {
    LOG(INFO) << "  S[" << i << "] = " << singular_values(i);
  }

  // 4) Normalize s_global by the median of |s|
  // Eigen::VectorXd s_abs = s_global.cwiseAbs();
  // std::vector<double> nonzero;
  // nonzero.reserve(num_edges);
  // for (int e = 0; e < num_edges; ++e) {
  //   if (s_abs(e) > 1e-9) {
  //     nonzero.push_back(s_abs(e));
  //   }
  // }

  // double scale_norm = 1.0;
  // if (!nonzero.empty()) {
  //   std::nth_element(nonzero.begin(),
  //                    nonzero.begin() + nonzero.size() / 2,
  //                    nonzero.end());
  //   scale_norm = nonzero[nonzero.size() / 2];
  // }

  // if (scale_norm <= 0.0) {
  //   scale_norm = 1.0;
  // }
  // s_global /= scale_norm;

  // 5) Build edge scales and simple support counts.
  std::vector<int> edge_support(num_edges, 0);
  for (const auto& gc : gconstraints) {
    edge_support[gc.e_ij]++;
    edge_support[gc.e_jk]++;
    edge_support[gc.e_ik]++;
  }

  int num_edges_before = num_edges;
  int num_edges_after  = 0;
  int num_edges_removed_by_scale = 0;

  for (const auto& kv : edge_to_id) {
    const uint64_t ekey = kv.first;
    const int eid = kv.second;
    const double s_val = std::abs(s_global(eid));

    if (s_val < kMinEdgeScale) {
      ++num_edges_removed_by_scale;
      continue;
    }

    EdgeScaleSample sample;
    sample.s          = s_val;
    sample.inliers    = edge_support[eid];  // number of local constraints touching this edge
    sample.median_err = 0.0;               // unknown here; we leave it as 0

    edge_scales[ekey].push_back(sample);
    ++num_edges_after;
  }

  LOG(INFO) << "[GLOBAL_LS] edges (unique) before = " << num_edges_before
            << ", after scale-threshold = " << num_edges_after
            << ", removed_by_scale = " << num_edges_removed_by_scale
            << " (thres=" << kMinEdgeScale << ")";

  if (VLOG_IS_ON(2)) {
    for (const auto& kv : edge_scales) {
      const uint64_t key = kv.first;
      const image_t u = static_cast<image_t>(key >> 32);
      const image_t v = static_cast<image_t>(key & 0xffffffff);
      const auto& vec = kv.second;
      double s_val = vec.front().s;
      LOG(INFO) << "[GLOBAL_LS] edge (" << u << "," << v
                << ") s = " << s_val
                << ", support = " << vec.front().inliers;
    }
  }
}

bool GlobalPositioner::GetWorldDirection(
      const ViewGraph& view_graph,
      const std::unordered_map<image_t, Image>& images,
      image_t src, image_t dst,
      Eigen::Vector3d* dir_world) const {
  Eigen::Matrix3d Rsd;
  Eigen::Vector3d t_hat_sd;
  if (!GetRelPose_I_to_J(view_graph, src, dst, &Rsd, &t_hat_sd)) {
    return false;
  }
  const auto it_dst = images.find(dst);
  if (it_dst == images.end()) return false;
  const Eigen::Matrix3d Rwc =
      it_dst->second.cam_from_world.rotation.inverse().toRotationMatrix();
  *dir_world = NormalizeSafe(Rwc * t_hat_sd);
  return true;
}

void GlobalPositioner::InitializeCamerasFromTriScales(
      const ViewGraph& view_graph,
      std::unordered_map<image_t, Image>& images,
      const std::unordered_map<uint64_t, std::vector<EdgeScaleSample>>& edge_scales) {
  if (images.empty()) return;

  // ------------------------------------------------------------------
  // 1) Build adjacency using only edges that have triplet-based scales.
  //    This defines the "good" graph on which we build a backbone.
  // ------------------------------------------------------------------
  using Neighbor = std::pair<image_t, double>;  // (neighbor id, best scale)
  std::unordered_map<image_t, std::vector<Neighbor>> good_adj;
  good_adj.reserve(images.size());

  for (const auto& [pair_id, ipair] : view_graph.image_pairs) {
    if (!ipair.is_valid) continue;

    const image_t u = ipair.image_id1;
    const image_t v = ipair.image_id2;
    const uint64_t key = EdgeKey(u, v);

    auto it_s = edge_scales.find(key);
    if (it_s == edge_scales.end() || it_s->second.empty()) {
      // No triplet-based scale for this edge -> do not use it in the backbone.
      continue;
    }

    const auto& samples = it_s->second;

    // Select the best sample based on (inliers, -median_err).
    const EdgeScaleSample* best = &samples[0];
    for (const auto& es : samples) {
      if (es.inliers > best->inliers) {
        best = &es;
      } else if (es.inliers == best->inliers &&
                 es.median_err < best->median_err) {
        best = &es;
      }
    }

    const double s_uv = best->s;
    if (s_uv <= 0.0) {
      continue;  // Just in case, ignore non-positive scales.
    }

    good_adj[u].emplace_back(v, s_uv);
    good_adj[v].emplace_back(u, s_uv);
  }

  // ------------------------------------------------------------------
  // 2) BFS over the "good" graph to initialize a backbone of cameras.
  //    Only edges with triplet scales contribute here (no 1.0 fallback).
  // ------------------------------------------------------------------
  std::unordered_set<image_t> visited;
  visited.reserve(images.size());

  std::queue<image_t> q;

  int num_backbone_components = 0;
  int num_backbone_nodes = 0;

  // We may have multiple connected components in the good graph.
  for (auto& [image_id, image] : images) {
    if (visited.count(image_id)) continue;
    if (good_adj.find(image_id) == good_adj.end()) {
      // This image has no incident good edges; we'll handle it later.
      continue;
    }

    // Start a new backbone component from this image.
    ++num_backbone_components;

    // For each component, we set its root translation to zero.
    // Different components are not aligned to each other yet; BA will
    // handle the global gauge freedoms later.
    image.cam_from_world.translation.setZero();
    visited.insert(image_id);
    q.push(image_id);
    ++num_backbone_nodes;

    while (!q.empty()) {
      const image_t u = q.front();
      q.pop();

      const auto it_adj = good_adj.find(u);
      if (it_adj == good_adj.end()) {
        continue;
      }

      for (const Neighbor& nb : it_adj->second) {
        const image_t v = nb.first;
        const double s_uv = nb.second;
        if (visited.count(v)) continue;

        Eigen::Vector3d dir_w;
        if (!GetWorldDirection(view_graph, images, u, v, &dir_w)) {
          // Fallback direction if the relative pose is missing or invalid.
          dir_w = Eigen::Vector3d(1, 0, 0);
        }

        images[v].cam_from_world.translation =
            images[u].cam_from_world.translation + s_uv * dir_w;

        visited.insert(v);
        q.push(v);
        ++num_backbone_nodes;
      }
    }
  }

  LOG(INFO) << "[TRI_RANSAC] backbone components (good edges only) = "
            << num_backbone_components
            << ", backbone cameras = " << num_backbone_nodes;

  // ------------------------------------------------------------------
  // 3) Attach remaining images (not in 'visited') to the backbone using
  //    the full view-graph. Here we allow 1.0-scale fallback, but only
  //    for nodes that could not be initialized from triplet-based edges.
  //
  //    We iteratively grow a tree from the set of 'visited' cameras:
  //    in each iteration, we try to connect any unvisited node that has
  //    at least one neighbor already in 'visited'.
  // ------------------------------------------------------------------
  const double kFallbackScale = 1.0;

  bool progress = true;
  int num_attached = 0;

  while (progress) {
    progress = false;

    for (auto& [image_id, image] : images) {
      if (visited.count(image_id)) {
        continue;  // Already initialized in backbone or attached.
      }

      // Find a neighbor in the view graph that is already visited.
      image_t best_neighbor = static_cast<image_t>(-1);

      // NOTE: This is O(#image_pairs * #images) in the worst case.
      //       For large problems, consider pre-building a full adjacency
      //       list for the view-graph to make this faster.
      for (const auto& [pair_id, ipair] : view_graph.image_pairs) {
        if (!ipair.is_valid) continue;
        const image_t u = ipair.image_id1;
        const image_t v = ipair.image_id2;

        if (u == image_id && visited.count(v)) {
          best_neighbor = v;
          break;
        } else if (v == image_id && visited.count(u)) {
          best_neighbor = u;
          break;
        }
      }

      if (best_neighbor == static_cast<image_t>(-1)) {
        // No connection to the current visited set in this iteration.
        continue;
      }

      Eigen::Vector3d dir_w;
      if (!GetWorldDirection(view_graph, images, best_neighbor, image_id, &dir_w)) {
        dir_w = Eigen::Vector3d(1, 0, 0);
      }

      images[image_id].cam_from_world.translation =
          images[best_neighbor].cam_from_world.translation +
          kFallbackScale * dir_w;

      visited.insert(image_id);
      ++num_attached;
      progress = true;
    }
  }

  const size_t num_total = images.size();
  const size_t num_uninitialized = num_total - visited.size();

  LOG(INFO) << "[TRI_RANSAC] attached cameras (fallback tree) = "
            << num_attached
            << ", total initialized = " << visited.size()
            << " / " << num_total
            << ", still uninitialized = " << num_uninitialized;
}

void GlobalPositioner::InitializePointsFromCameras(
      std::unordered_map<camera_t, Camera>& cameras,
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks) {
  (void)cameras;  // not used
  constexpr double kInitDepth = 5.0;
  for (auto& [tid, tr] : tracks) {
    if (tr.observations.empty()) continue;
    const image_t img_id = tr.observations.front().first;
    const int feat_id = static_cast<int>(tr.observations.front().second);

    auto it_img = images.find(img_id);
    if (it_img == images.end()) continue;
    if (!it_img->second.is_registered) continue;

    const Eigen::Vector3d ray_c =
        NormalizeSafe(it_img->second.features_undist[feat_id]);
    const Eigen::Vector3d Cw = it_img->second.cam_from_world.translation;
    const Eigen::Matrix3d Rwc =
        it_img->second.cam_from_world.rotation.inverse().toRotationMatrix();

    tracks[tid].xyz = Cw + kInitDepth * (Rwc * ray_c);
    tracks[tid].is_initialized = true;
  }
}

}  // namespace glomap
