#include "glomap/estimators/global_positioning.h"

#include "glomap/estimators/cost_function.h"
#include "glomap/estimators/cost_function_scaled.h"
#include <queue>

#include <ceres/ceres.h>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <limits>

namespace glomap {
namespace {

Eigen::Vector3d RandVector3d(std::mt19937& random_generator,
                             double low,
                             double high) {
  std::uniform_real_distribution<double> distribution(low, high);
  return Eigen::Vector3d(distribution(random_generator),
                         distribution(random_generator),
                         distribution(random_generator));
}

// double RandomDouble(std::mt19937& random_generator,
//                     double min,
//                     double max) {
//     std::uniform_real_distribution<double> dist(min, max);
//     return dist(random_generator);

// }

static std::vector<image_t> SelectTopKImagesByObs(
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<track_t, Track>& tracks,
    size_t K) {
  std::unordered_map<image_t, size_t> obs_count;
  for (const auto& [tid, tr] : tracks) {
    for (const auto& obs : tr.observations) obs_count[obs.first]++;
  }
  std::vector<std::pair<image_t, size_t>> v(obs_count.begin(), obs_count.end());
  std::sort(v.begin(), v.end(), [](auto& a, auto& b){ return a.second > b.second; });
  std::vector<image_t> out;
  out.reserve(std::min(K, v.size()));
  for (size_t i = 0; i < std::min(K, v.size()); ++i) out.push_back(v[i].first);
  return out;
}

class SScalarTraceCallback : public ceres::IterationCallback {
 public:
  SScalarTraceCallback(const std::vector<double>* s_vars,
                       const std::unordered_map<image_t, size_t>* s_index,
                       std::vector<image_t> watch_ids,
                       const std::string& csv_path,
                       int log_every_n = 1)
      : s_vars_(s_vars),
        s_index_(s_index),
        watch_ids_(std::move(watch_ids)),
        csv_(csv_path, std::ios::out),
        log_every_n_(log_every_n) {
    // CSV header
    csv_ << "iter,cost,grad_norm,step_norm,min_s,max_s,mean_s,std_s";
    for (auto id : watch_ids_) csv_ << ",s[" << id << "]";
    csv_ << "\n";
  }

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
    if (summary.iteration == 0 || (summary.iteration % log_every_n_) == 0) {
      // Basic stats on all s
      double min_s = std::numeric_limits<double>::infinity();
      double max_s = -std::numeric_limits<double>::infinity();
      double mean_s = 0.0;
      const size_t n = s_vars_->size();
      for (size_t i = 0; i < n; ++i) {
        double v = (*s_vars_)[i];
        min_s = std::min(min_s, v);
        max_s = std::max(max_s, v);
        mean_s += v;
      }
      mean_s /= std::max<size_t>(1, n);
      // stddev
      double var = 0.0;
      for (size_t i = 0; i < n; ++i) {
        double d = (*s_vars_)[i] - mean_s;
        var += d * d;
      }
      double std_s = (n > 1) ? std::sqrt(var / (n - 1)) : 0.0;

      // Row: iteration summary + stats
      csv_ << summary.iteration << ","
           << summary.cost << ","
           << summary.gradient_norm << ","
           << summary.step_norm << ","
           << min_s << ","
           << max_s << ","
           << mean_s << ","
           << std_s;

      // Selected s[image_id]
      for (auto id : watch_ids_) {
        auto it = s_index_->find(id);
        double sval = (it == s_index_->end()) ? std::numeric_limits<double>::quiet_NaN()
                                              : (*s_vars_)[it->second];
        csv_ << "," << sval;
      }
      csv_ << "\n";
      csv_.flush();
    }
    // Continue solving
    return ceres::CallbackReturnType::SOLVER_CONTINUE;
  }

 private:
  const std::vector<double>* s_vars_;
  const std::unordered_map<image_t, size_t>* s_index_;
  std::vector<image_t> watch_ids_;
  std::ofstream csv_;
  int log_every_n_;
};

}  // namespace

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

  LOG(INFO) << "Setting up the global positioner problem";

  // Setup the problem.
  SetupProblem(view_graph, tracks);

  // Initialize camera translations to be random.
  // Also, convert the camera pose translation to be the camera center.
  InitializeRandomPositions(view_graph, images, tracks);

  // Add the camera to camera constraints to the problem.
  if (options_.constraint_type != GlobalPositionerOptions::ONLY_POINTS &&
      options_.constraint_type != GlobalPositionerOptions::ONLY_SCALEDPOINTS) {
    AddCameraToCameraConstraints(view_graph, images);
  }

  // Add the point to camera constraints to the problem.
  if (options_.constraint_type != GlobalPositionerOptions::ONLY_CAMERAS) {
    AddPointToCameraConstraints(cameras, images, tracks, view_graph);
  }

  AddCamerasAndPointsToParameterGroups(images, tracks);

  // Parameterize the variables, set image poses / tracks / scales to be
  // constant if desired
  ParameterizeVariables(images, tracks);

  LOG(INFO) << "Solving the global positioner problem";

  ceres::Solver::Summary summary;
  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);

  // Pick what to watch (e.g., top 8 by obs)
  std::vector<image_t> watch_ids;
  if (options_.constraint_type == GlobalPositionerOptions::ONLY_SCALEDPOINTS) {
    watch_ids = SelectTopKImagesByObs(images, tracks, 8);
    // or: watch_ids = {123, 456, ...};
    options_.solver_options.update_state_every_iteration = true;
    auto* cb = new SScalarTraceCallback(&s_vars_, &s_index_, watch_ids,
                                        "s_trace.csv", /*log_every_n=*/1);
    options_.solver_options.callbacks.push_back(cb); // Ceres takes ownership
  }

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
  // Allocate enough memory for the scales. One for each residual.
  // Due to possibly invalid image pairs or tracks, the actual number of
  // residuals may be smaller.
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
    if (image_pair.is_valid == false) continue;

    constrained_positions.insert(image_pair.image_id1);
    constrained_positions.insert(image_pair.image_id2);
  }

  if (options_.constraint_type != GlobalPositionerOptions::ONLY_CAMERAS) {
    for (const auto& [track_id, track] : tracks) {
      if (track.observations.size() < options_.min_num_view_per_track) continue;
      for (const auto& observation : tracks[track_id].observations) {
        if (images.find(observation.first) == images.end()) continue;
        Image& image = images[observation.first];
        if (!image.is_registered) continue;
        constrained_positions.insert(observation.first);
      }
    }
  }

  if (!options_.generate_random_positions || !options_.optimize_positions
      || options_.constraint_type == GlobalPositionerOptions::ONLY_SCALEDPOINTS) {
    for (auto& [image_id, image] : images) {
      image.cam_from_world.translation = image.Center();
    }
    return;
  }

  // Generate random positions for the cameras centers.
  for (auto& [image_id, image] : images) {
    // Only set the cameras to be random if they are needed to be optimized
    if (constrained_positions.find(image_id) != constrained_positions.end())
      image.cam_from_world.translation =
          100.0 * RandVector3d(random_generator_, -1, 1);
    else
      image.cam_from_world.translation = image.Center();
  }

  VLOG(2) << "Constrained positions: " << constrained_positions.size();
}

void GlobalPositioner::AddCameraToCameraConstraints(
    const ViewGraph& view_graph, std::unordered_map<image_t, Image>& images) {
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) continue;

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
        -(images[image_id2].cam_from_world.rotation.inverse() *
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
    std::unordered_map<track_t, Track>& tracks,
    const ViewGraph& view_graph) {
  // The number of camera-to-camera constraints coming from the relative poses

  const size_t num_cam_to_cam = problem_->NumResidualBlocks();
  // Find the tracks that are relevant to the current set of cameras
  const size_t num_pt_to_cam = tracks.size();

  VLOG(2) << num_pt_to_cam
          << " point to camera constriants were added to the position "
             "estimation problem.";

  if (num_pt_to_cam == 0) return;

  double weight_scale_pt = 1.0;
  // Set the relative weight of the point to camera constraints based on
  // the number of camera to camera constraints.
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

  // I want ONLY_SCALEDPOINTS option to guide the process as the same as the ONLY_POINTS
  // except for this process --- chosing between AddTrackToProblem and AddTrackToScaledCamProblem
  if (options_.constraint_type != GlobalPositionerOptions::ONLY_SCALEDPOINTS){
  for (auto& [track_id, track] : tracks) {
    if (track.observations.size() < options_.min_num_view_per_track) continue;

    // Only set the points to be random if they are needed to be optimized
    if (options_.optimize_points && options_.generate_random_points) {
      track.xyz = 100.0 * RandVector3d(random_generator_, -1, 1);
      track.is_initialized = true;
    }

    AddTrackToProblem(track_id, cameras, images, tracks);
    }
  } else {
    // 1) fixed world directions
    image_t root_id = images.begin()->first;
    BuildScaledCamDirectionsTree(view_graph, images, root_id);
    // 2) set global root center (choose one policy)
    // simplest: zero
    c_root_fixed_.setZero();
    // 3) preallocate s_i with stable addresses, init to 1.0
    s_vars_.clear();
    s_index_.clear();
    s_vars_.assign(images.size(), 1.0);
    // s_vars_.resize(images.size());            // fixed capacity & addresses
    // for (auto &v : s_vars_){
    //   v = 100.0 * RandomDouble(random_generator_, -1, 1);
    // }
    {
      size_t idx = 0;
      for (const auto& [image_id, image] : images) {
        s_index_[image_id] = idx++;
      }
    }
    // 4) add residuals
    for (auto& [track_id, track] : tracks) {
      if (track.observations.size() < options_.min_num_view_per_track) continue;

      if (options_.optimize_points && options_.generate_random_points) {
        track.xyz = 100.0 * RandVector3d(random_generator_, -1, 1);
        track.is_initialized = true;
      }
      AddTrackToScaledCamProblem(track_id, cameras, images, tracks);
    }
  }
}

void GlobalPositioner::AddTrackToScaledCamProblem(
    track_t track_id,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {

  for (const auto& obs : tracks[track_id].observations) {
    const image_t image_id = obs.first;
    if (!images.count(image_id)) continue;

    Image& image = images[image_id];
    if (!image.is_registered) continue;

    const Eigen::Vector3d& f = image.features_undist[obs.second];
    if (f.array().isNaN().any()) continue;

    // Convert observation ray to world coordinates
    Eigen::Vector3d v_obs_world = image.cam_from_world.rotation.inverse() * f;
    if (v_obs_world.norm() > 1e-12) v_obs_world.normalize();

    // Get fixed direction vector for this camera
    auto it_dir = dir_param_holder_.find(image_id);
    if (it_dir == dir_param_holder_.end()) {
      LOG(FATAL) << "Direction missing for image_id=" << image_id
                << ". BuildScaledCamDirections must cover all cameras.";
    }
    const Eigen::Vector3d& dir_fixed = it_dir->second;

    ceres::CostFunction* cost = PtCamDirErrorScaledRoot::Create(v_obs_world, c_root_fixed_, dir_fixed);

    // Prepare scale variable for this camera if not already created
    double* s = &s_vars_[s_index_.at(image_id)];

    // Prepare observation-specific scale parameter d_ik
    CHECK_GT(scales_.capacity(), scales_.size())
        << "Not enough capacity was reserved for the scales.";
    double& d_ik = scales_.emplace_back(1.0);

    // Select appropriate loss function
    ceres::LossFunction* loss =
        cameras[image.camera_id].has_prior_focal_length
            ? loss_function_ptcam_calibrated_.get()
            : loss_function_ptcam_uncalibrated_.get();

    // Residual block parameters: [C0(3)], [X(3)], [s(1)], [dir(3 const)], [d_ik(1)]
    problem_->AddResidualBlock(cost, loss,
                              tracks[track_id].xyz.data(),  // X
                              s,                             // s
                              &d_ik);                        // d_ik

    // Fix base camera center and direction vector (constants)
    problem_->SetParameterLowerBound(&d_ik, 0, 1e-5);
  }
}

void GlobalPositioner::BuildScaledCamDirectionsTree(
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    image_t root_id) {
  // (1) Build adjacency list from valid image pairs
  std::unordered_map<image_t, std::vector<image_t>> adj;
  for (const auto& [pair_id, p] : view_graph.image_pairs) {
    if (!p.is_valid) continue;
    if (!images.count(p.image_id1) || !images.count(p.image_id2)) continue;
    adj[p.image_id1].push_back(p.image_id2);
    adj[p.image_id2].push_back(p.image_id1);
  }

  // (2) BFS traversal from the root to assign parent relationships
  std::unordered_map<image_t, image_t> parent;
  std::queue<image_t> q;
  parent[root_id] = root_id;  // root is its own parent
  q.push(root_id);

  while (!q.empty()) {
    image_t u = q.front(); q.pop();
    if (!adj.count(u)) continue;
    for (image_t v : adj[u]) {
      if (parent.find(v) != parent.end()) continue;  // already visited
      parent[v] = u;
      q.push(v);
    }
  }

  // (3) For each node, assign dir_i as the world direction of parent->i edge
  dir_param_holder_.clear();

  for (const auto& [i, img] : images) {
    if (i == root_id || parent.find(i) == parent.end()) {
      // Root or isolated node: assign arbitrary axis (X-axis)
      dir_param_holder_[i] = Eigen::Vector3d(1,0,0);
      continue;
    }
    image_t par = parent[i];

    // Compute world direction directly using the available pair orientation
    Eigen::Vector3d u_world;
    bool found = false;

    for (const auto& [pair_id, P] : view_graph.image_pairs) {
      if (!P.is_valid) continue;

      if (P.image_id1 == par && P.image_id2 == i) {
        // Case: par(cam1) -> i(cam2). t is in camera i frame.
        // u_world = -(R_i^T) * t_{i<-par}
        const Eigen::Matrix3d R_i_T   = images.at(i).cam_from_world.rotation.toRotationMatrix().transpose();


        u_world = -(R_i_T * P.cam2_from_cam1.translation);
        found = true;
        break;
      }
      if (P.image_id1 == i && P.image_id2 == par) {
        // Case: i(cam1) -> par(cam2). t is in camera par frame.
        // We want direction of (C_i - C_par) in world:
        // C_par - C_i ~ -(R_par^T) * t_{par<-i}  ->  C_i - C_par ~ +(R_par^T) * t_{par<-i}
        const Eigen::Matrix3d R_par_T = images.at(par).cam_from_world.rotation.toRotationMatrix().transpose();
        u_world = R_par_T * P.cam2_from_cam1.translation;
        found = true;
        break;
      }
    }

    if (!found) {
      // If no valid pair exists, fallback to arbitrary axis
      u_world = Eigen::Vector3d(1,0,0);
    } else {
      if (u_world.norm() > 1e-12) u_world.normalize();
      else u_world = Eigen::Vector3d(1,0,0);
    }

    dir_param_holder_[i] = u_world;
  }
}

void GlobalPositioner::AddTrackToProblem(
    track_t track_id,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  // For each view in the track add the point to camera correspondences.
  for (const auto& observation : tracks[track_id].observations) {
    if (images.find(observation.first) == images.end()) continue;

    Image& image = images[observation.first];
    if (!image.is_registered) continue;

    const Eigen::Vector3d& feature_undist =
        image.features_undist[observation.second];
    if (feature_undist.array().isNaN().any()) {
      LOG(WARNING)
          << "Ignoring feature because it failed to undistort: track_id="
          << track_id << ", image_id=" << observation.first
          << ", feature_id=" << observation.second;
      continue;
    }

    const Eigen::Vector3d translation =
        image.cam_from_world.rotation.inverse() *
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

    // For calibrated and uncalibrated cameras, use different loss functions
    // Down weight the uncalibrated cameras
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
  // Create a custom ordering for Schur-based problems.
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();

  // Add scale parameters to group 0 (large and independent)
  for (double& scale : scales_) {
    parameter_ordering->AddElementToGroup(&scale, 0);
  }

  // Add point parameters to group 1.
  int group_id = 1;
  if (tracks.size() > 0) {
    for (auto& [track_id, track] : tracks) {
      if (problem_->HasParameterBlock(track.xyz.data()))
        parameter_ordering->AddElementToGroup(track.xyz.data(), group_id);
    }
    group_id++;
  }

  // Add camera parameters to group 2 if there are tracks, otherwise group 1.
  for (auto& [image_id, image] : images) {
    if (problem_->HasParameterBlock(image.cam_from_world.translation.data())) {
      parameter_ordering->AddElementToGroup(
          image.cam_from_world.translation.data(), group_id);
    }
  }

  if (options_.constraint_type == GlobalPositionerOptions::ONLY_SCALEDPOINTS) {
    for (const auto& [image_id, idx] : s_index_) {
      double* s = &s_vars_[idx];
      if (problem_->HasParameterBlock(s)) {
        parameter_ordering->AddElementToGroup(s, group_id); // same as cameras
      }
    }
  }
}

void GlobalPositioner::ParameterizeVariables(
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  // For the global positioning, do not set any camera to be constant for easier
  // convergence

  // If do not optimize the positions, set the camera positions to be constant
  if (!options_.optimize_positions) {
    for (auto& [image_id, image] : images)
      if (problem_->HasParameterBlock(image.cam_from_world.translation.data()))
        problem_->SetParameterBlockConstant(
            image.cam_from_world.translation.data());
  }

  // If do not optimize the rotations, set the camera rotations to be constant
  if (!options_.optimize_points) {
    for (auto& [track_id, track] : tracks) {
      if (problem_->HasParameterBlock(track.xyz.data())) {
        problem_->SetParameterBlockConstant(track.xyz.data());
      }
    }
  }

  // If do not optimize the scales, set the scales to be constant
  if (!options_.optimize_scales) {
    for (double& scale : scales_) {
      problem_->SetParameterBlockConstant(&scale);
    }
  }

  // Set up the options for the solver
  // Do not use iterative solvers, for its suboptimal performance.
  if (tracks.size() > 0) {
    options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    options_.solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
  } else {
    options_.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options_.solver_options.preconditioner_type = ceres::JACOBI;
  }

  if (options_.constraint_type == GlobalPositionerOptions::ONLY_SCALEDPOINTS) {
    for (auto& [image_id, image] : images) {
      if (problem_->HasParameterBlock(image.cam_from_world.translation.data())) {
        problem_->SetParameterBlockConstant(image.cam_from_world.translation.data());
      }
    }
  }
}

void GlobalPositioner::ConvertResults(
    std::unordered_map<image_t, Image>& images) {
  // translation now stores the camera position, needs to convert back to
  // translation
  for (auto& [image_id, image] : images) {
    image.cam_from_world.translation =
        -(image.cam_from_world.rotation * image.cam_from_world.translation);
  }
}

}  // namespace glomap
