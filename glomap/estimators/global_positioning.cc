#include "glomap/estimators/global_positioning.h"

#include "glomap/estimators/cost_function.h"
#include "glomap/estimators/cost_function_scaled.h"
#include <queue>

#include <ceres/ceres.h>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <limits>
#include <iomanip>

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

// Get world edge direction for (u -> v) from view_graph and image rotations.
// Returns unit vector in dir_edge_world. False if not found/degenerate.
static bool GetEdgeWorldDirection(
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    image_t u, image_t v,
    Eigen::Vector3d* dir_edge_world /*out*/) {

  for (const auto& [pair_id, P] : view_graph.image_pairs) {
    if (!P.is_valid) continue;

    if (P.image_id1 == u && P.image_id2 == v) {
      // u(cam1) -> v(cam2); translation given in camera v frame
      const Eigen::Matrix3d RvT =
          images.at(v).cam_from_world.rotation.toRotationMatrix().transpose();
      *dir_edge_world = -(RvT * P.cam2_from_cam1.translation);
      const double n = dir_edge_world->norm();
      if (n > 1e-12) { *dir_edge_world /= n; return true; }
      return false;
    }
    if (P.image_id1 == v && P.image_id2 == u) {
      // v(cam1) -> u(cam2); need opposite direction for u->v
      const Eigen::Matrix3d RuT =
          images.at(u).cam_from_world.rotation.toRotationMatrix().transpose();
      *dir_edge_world =  (RuT * P.cam2_from_cam1.translation);
      const double n = dir_edge_world->norm();
      if (n > 1e-12) { *dir_edge_world /= n; return true; }
      return false;
    }
  }
  return false;
}

// Build a Maximum Spanning Tree (Prim) given an adjacency with integer weights.
// `adj[u]` holds (v, weight). Returns parent map, parent[root]=root.
static std::unordered_map<image_t, image_t> BuildMST_MaxWeight(
    const std::unordered_map<image_t, std::vector<std::pair<image_t,int>>>& adj,
    const std::unordered_map<image_t, Image>& images,
    image_t root_id) {

  using Item = std::tuple<int, image_t, image_t>; // (weight, u, v)
  auto cmp = [](const Item& a, const Item& b){ return std::get<0>(a) < std::get<0>(b); };
  std::priority_queue<Item, std::vector<Item>, decltype(cmp)> pq(cmp);

  std::unordered_map<image_t, image_t> parent;
  std::unordered_set<image_t> in_mst;
  parent.reserve(images.size());
  in_mst.reserve(images.size());

  parent[root_id] = root_id;
  in_mst.insert(root_id);
  if (adj.count(root_id)) {
    for (auto& [v, w] : adj.at(root_id)) pq.emplace(w, root_id, v);
  }

  while (!pq.empty() && parent.size() < images.size()) {
    auto [w, u, v] = pq.top(); pq.pop();
    if (in_mst.count(v)) continue;
    parent[v] = u;
    in_mst.insert(v);
    if (adj.count(v)) {
      for (auto& [to, wt] : adj.at(v)) {
        if (!in_mst.count(to)) pq.emplace(wt, v, to);
      }
    }
  }

  // Any image missing (disconnected) -> parent to itself
  for (const auto& [img_id, _] : images) {
    if (!parent.count(img_id)) parent[img_id] = img_id;
  }
  return parent;
}

static image_t SelectRootByInlierSum(const ViewGraph& view_graph,
                                     const std::unordered_map<image_t, Image>& images) {
  // Accumulate inlier sums per image
  std::unordered_map<image_t, long long> sum_inliers;
  sum_inliers.reserve(images.size());

  for (const auto& [pair_id, P] : view_graph.image_pairs) {
    if (!P.is_valid) continue;
    if (!images.count(P.image_id1) || !images.count(P.image_id2)) continue;
    const size_t cnt = P.inliers.size();
    sum_inliers[P.image_id1] += static_cast<long long>(cnt);
    sum_inliers[P.image_id2] += static_cast<long long>(cnt);
  }

  // If no inlier info, fallback to the first image
  if (images.empty()) {
    return 0; // undefined; caller should guard earlier
  }
  image_t fallback = images.begin()->first;

  if (sum_inliers.empty()) return fallback;

  // Argmax over images present in 'images'
  image_t best = fallback;
  long long best_sum = std::numeric_limits<long long>::min();
  for (const auto& [img_id, _img] : images) {
    const long long s = sum_inliers.count(img_id) ? sum_inliers[img_id] : 0LL;
    if (s > best_sum) {
      best_sum = s;
      best = img_id;
    }
  }
  return best;
}

// Compute midpoint triangulation from two rays (Ca + ta*ra, Cb + tb*rb).
// Returns the midpoint of the shortest segment between the two skew lines.
// ra, rb must be unit vectors.
static inline Eigen::Vector3d TriangulateMidpoint(
    const Eigen::Vector3d& Ca, const Eigen::Vector3d& ra,
    const Eigen::Vector3d& Cb, const Eigen::Vector3d& rb) {

  const Eigen::Vector3d w0 = Ca - Cb;
  const double a = ra.dot(ra);           // = 1 if ra is unit
  const double b = ra.dot(rb);
  const double c = rb.dot(rb);           // = 1 if rb is unit
  const double d = ra.dot(w0);
  const double e = rb.dot(w0);
  const double denom = a*c - b*b;

  double ta, tb;
  if (std::abs(denom) < 1e-12) {
    // Nearly parallel rays, fallback to simple average along ra
    ta = -d / std::max(1e-12, a);
    tb =  0.0;
  } else {
    ta = (b*e - c*d) / denom;
    tb = (a*e - b*d) / denom;
  }
  const Eigen::Vector3d Pa = Ca + ta * ra;
  const Eigen::Vector3d Pb = Cb + tb * rb;
  return 0.5 * (Pa + Pb);
}

// Check if 3D point Xw is inside camera frustum defined in normalized coords.
// We use the candidate camera center Ccand (not images[].translation) because
// centers are parameterized as C = C_root + s_i * dir_i at init stage.
static inline bool InFrustumNormalized(
    const Eigen::Vector3d& Xw,
    const Eigen::Vector3d& Ccand,
    const Eigen::Quaterniond& R_cw,     // camera-from-world rotation
    double max_norm_xy,                  // e.g., 1.5 ~ 2.0
    double min_z                         // e.g., 1e-3
) {
  // Xc = R_cw * (Xw - C)
  const Eigen::Vector3d Xc = R_cw * (Xw - Ccand);
  if (Xc.z() <= min_z) return false;
  const double nx = Xc.x() / Xc.z();
  const double ny = Xc.y() / Xc.z();
  return (std::abs(nx) < max_norm_xy) && (std::abs(ny) < max_norm_xy);
}

// Collect world rays for tracks observed by both images u and v.
// Returns vector of pairs (r_u, r_v), both unit vectors in world coordinates.
static std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
CollectSharedWorldRays(image_t u, image_t v,
                       const std::unordered_map<image_t, Image>& images,
                       const std::unordered_map<track_t, Track>& tracks) {
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> rays;
  rays.reserve(256);

  const auto& img_u = images.at(u);
  const auto& img_v = images.at(v);
  if (!img_u.is_registered || !img_v.is_registered) return rays;

  const Eigen::Matrix3d RuT = img_u.cam_from_world.rotation.toRotationMatrix().transpose();
  const Eigen::Matrix3d RvT = img_v.cam_from_world.rotation.toRotationMatrix().transpose();

  for (const auto& [tid, tr] : tracks) {
    int idx_u = -1, idx_v = -1;
    for (const auto& ob : tr.observations) {
      if (ob.first == u) idx_u = static_cast<int>(ob.second);
      else if (ob.first == v) idx_v = static_cast<int>(ob.second);
      if (idx_u >= 0 && idx_v >= 0) break;
    }
    if (idx_u < 0 || idx_v < 0) continue;

    const Eigen::Vector3d& fu = img_u.features_undist[idx_u];
    const Eigen::Vector3d& fv = img_v.features_undist[idx_v];
    if (fu.array().isNaN().any() || fv.array().isNaN().any()) continue;

    Eigen::Vector3d ru = RuT * fu;
    Eigen::Vector3d rv = RvT * fv;
    const double nu = ru.norm(), nv = rv.norm();
    if (nu <= 1e-12 || nv <= 1e-12) continue;
    ru /= nu; rv /= nv;

    rays.emplace_back(ru, rv);
  }
  return rays;
}

// Pick s_v by frustum voting on N sampled scales.
// Returns the chosen s_v; early-returns the first candidate with >= M_thresh votes.
// Otherwise returns the candidate with the maximum votes.
// Returns chosen s_v; sets *used_fallback=true if we returned a fallback
// (empty rays or no candidate reached M_thresh).
static double PickSvByFrustum(
    image_t u, image_t v,
    double s_u,
    const Eigen::Vector3d& dir_u,
    const Eigen::Vector3d& dir_v,
    const Eigen::Vector3d& C_root,
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<track_t, Track>& tracks,
    int    N_samples,
    double span,
    int    M_thresh,
    double max_norm_xy,
    double min_z,
    bool   require_both_cams,
    bool*  used_fallback
) {
  // default to fallback unless we early-accept a good candidate
  if (used_fallback) *used_fallback = true;

  auto rays = CollectSharedWorldRays(u, v, images, tracks);
  if (rays.empty()) {
    return s_u;  // fallback: no shared rays
  }

  const auto& img_u = images.at(u);
  const auto& img_v = images.at(v);
  const Eigen::Quaterniond& Rcw_u = img_u.cam_from_world.rotation;
  const Eigen::Quaterniond& Rcw_v = img_v.cam_from_world.rotation;

  const Eigen::Vector3d C_u = C_root + s_u * dir_u;

  if (N_samples < 3) N_samples = 3;
  if ((N_samples % 2) == 0) N_samples += 1;
  if (span <= 0.0) span = 1.0;

  const double half = 0.5 * (N_samples - 1);
  const double step = span / static_cast<double>(N_samples - 1);

  double best_s     = s_u;
  int    best_votes = -1;

  for (int k = 0; k < N_samples; ++k) {
    const double offset = (static_cast<double>(k) - half) * step;
    const double s_v    = s_u + offset;
    const Eigen::Vector3d C_v = C_root + s_v * dir_v;

    int votes = 0;
    for (const auto& rv_pair : rays) {
      Eigen::Vector3d r_u = rv_pair.first;
      Eigen::Vector3d r_v = rv_pair.second;
      const double nu = r_u.norm(), nv = r_v.norm();
      if (nu <= 1e-12 || nv <= 1e-12) continue;
      r_u /= nu; r_v /= nv;
      if (r_u.cross(r_v).squaredNorm() < 1e-8) continue;

      const Eigen::Vector3d X = TriangulateMidpoint(C_u, r_u, C_v, r_v);
      const bool ok_v = InFrustumNormalized(X, C_v, Rcw_v, max_norm_xy, min_z);
      bool ok = ok_v;
      if (require_both_cams) {
        const bool ok_u = InFrustumNormalized(X, C_u, Rcw_u, max_norm_xy, min_z);
        ok = ok_u && ok_v;
      }
      votes += (ok ? 1 : 0);
    }

    if (votes >= M_thresh) {
      if (used_fallback) *used_fallback = false;  // accepted a strong candidate
      return s_v;
    }
    if (votes > best_votes) {
      best_votes = votes;
      best_s     = s_v;
    }
  }

  // fallback: no candidate reached M_thresh
  return best_s;
}

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
    // Initial root
    // image_t root_id = images.begin()->first;

    // Set the node with largest inliers as a root
    image_t root_id = SelectRootByInlierSum(view_graph, images);

    // build per-camera world directions (rooted)
    BuildScaledCamDirectionsTree(view_graph, images, root_id);

    // initialize s_i using MST weighted by match counts (preferred)
    // (If your ImagePair uses a different field than `num_inlier_matches`, rename there.)
    InitScalesByMST_FromViewGraphMatches(view_graph, images, tracks, root_id);

    DumpInitScalesCSV(cameras, images, "init_scales.csv");

    // 2) set global root center (choose one policy)
    // simplest: center
    c_root_fixed_ = images.at(root_id).Center();
    // 3) preallocate s_i with stable addresses, init to 1.0

    // Uniform 1 scale setting
    // s_vars_.clear();
    // s_index_.clear();
    // s_vars_.assign(images.size(), 1.0);

    // Random scale setting
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
    const ViewGraph& /*view_graph*/,
    const std::unordered_map<image_t, Image>& images,
    image_t /*root_id*/) {

  // We no longer rely on parent->child edge direction to define dir_i.
  // Instead, use each camera's optical axis expressed in world coordinates:
  // dir_i = R_wc * [0,0,1], where R_wc = (R_cw)^T.

  dir_param_holder_.clear();
  bool random_flip_dirs = true;

  const Eigen::Vector3d ez(0.0, 0.0, 1.0);  // camera forward in camera frame (z+)

  for (const auto& [img_id, img] : images) {
    // R_cw: world->camera;  R_wc = R_cw^T
    const Eigen::Matrix3d R_wc =
        img.cam_from_world.rotation.toRotationMatrix().transpose();

    // forward in world coordinates
    Eigen::Vector3d forward_w = R_wc * ez;

    // normalize (fallback to ez if degenerate)
    const double n = forward_w.norm();
    if (n > 1e-12) {
      forward_w /= n;
    } else {
      forward_w = ez;  // very unlikely, but keeps things defined
    }

    dir_param_holder_[img_id] = forward_w;
    if (random_flip_dirs) {
      static thread_local std::mt19937 rng(12345);
      std::bernoulli_distribution coin(0.5);
      if (coin(rng)) {
        dir_param_holder_[img_id] = -dir_param_holder_[img_id];
      }
    }

  }


  // NOTE:
  // - We intentionally do NOT flip/align sign using any pairwise edge direction.
  //   Doing so can bias all directions to point toward the root and cause collapse.
  // - If you ever want a *very mild* consistency rule, you could add a small
  //   heuristic here, but keep it optional and avoid global sign forcing.
}


// void GlobalPositioner::BuildScaledCamDirectionsTree(
//     const ViewGraph& view_graph,
//     const std::unordered_map<image_t, Image>& images,
//     image_t root_id) {
//   // (1) Build adjacency list from valid image pairs
//   std::unordered_map<image_t, std::vector<image_t>> adj;
//   for (const auto& [pair_id, p] : view_graph.image_pairs) {
//     if (!p.is_valid) continue;
//     if (!images.count(p.image_id1) || !images.count(p.image_id2)) continue;
//     adj[p.image_id1].push_back(p.image_id2);
//     adj[p.image_id2].push_back(p.image_id1);
//   }

//   // (2) BFS traversal from the root to assign parent relationships
//   std::unordered_map<image_t, image_t> parent;
//   std::queue<image_t> q;
//   parent[root_id] = root_id;  // root is its own parent
//   q.push(root_id);

//   while (!q.empty()) {
//     image_t u = q.front(); q.pop();
//     if (!adj.count(u)) continue;
//     for (image_t v : adj[u]) {
//       if (parent.find(v) != parent.end()) continue;  // already visited
//       parent[v] = u;
//       q.push(v);
//     }
//   }

//   // (3) For each node, assign dir_i as the world direction of parent->i edge
//   dir_param_holder_.clear();

//   for (const auto& [i, img] : images) {
//     if (i == root_id || parent.find(i) == parent.end()) {
//       // Root or isolated node: assign arbitrary axis (X-axis)
//       dir_param_holder_[i] = Eigen::Vector3d(1,0,0);
//       continue;
//     }
//     image_t par = parent[i];

//     // Compute world direction directly using the available pair orientation
//     Eigen::Vector3d u_world;
//     bool found = false;

//     for (const auto& [pair_id, P] : view_graph.image_pairs) {
//       if (!P.is_valid) continue;

//       if (P.image_id1 == par && P.image_id2 == i) {
//         // Case: par(cam1) -> i(cam2). t is in camera i frame.
//         // u_world = -(R_i^T) * t_{i<-par}
//         const Eigen::Matrix3d R_i_T   = images.at(i).cam_from_world.rotation.toRotationMatrix().transpose();


//         u_world = -(R_i_T * P.cam2_from_cam1.translation);
//         found = true;
//         break;
//       }
//       if (P.image_id1 == i && P.image_id2 == par) {
//         // Case: i(cam1) -> par(cam2). t is in camera par frame.
//         // We want direction of (C_i - C_par) in world:
//         // C_par - C_i ~ -(R_par^T) * t_{par<-i}  ->  C_i - C_par ~ +(R_par^T) * t_{par<-i}
//         const Eigen::Matrix3d R_par_T = images.at(par).cam_from_world.rotation.toRotationMatrix().transpose();
//         u_world = R_par_T * P.cam2_from_cam1.translation;
//         found = true;
//         break;
//       }
//     }

//     if (!found) {
//       // If no valid pair exists, fallback to arbitrary axis
//       u_world = Eigen::Vector3d(1,0,0);
//     } else {
//       if (u_world.norm() > 1e-12) u_world.normalize();
//       else u_world = Eigen::Vector3d(1,0,0);
//     }

//     dir_param_holder_[i] = u_world;
//   }
// }

// GlobalPositioner 클래스 메서드로 추가
void GlobalPositioner::DumpInitScalesCSV(
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, Image>& images,
    const std::string& csv_path) const
{
  std::ofstream csv(csv_path, std::ios::out);
  csv << "image_id,s_value,has_dir,dir_x,dir_y,dir_z,cx,cy,cz,cam_w,cam_h\n";

  for (const auto& kv : images) {
    const image_t img_id = kv.first;
    const auto&   img    = kv.second;

    // s 값
    double s_val = 0.0;
    auto it_s = s_index_.find(img_id);
    if (it_s != s_index_.end() && it_s->second < s_vars_.size())
      s_val = s_vars_[it_s->second];

    // dir
    bool has_dir = false;
    Eigen::Vector3d dir(0,0,0);
    auto it_d = dir_param_holder_.find(img_id);
    if (it_d != dir_param_holder_.end()) {
      dir = it_d->second;
      has_dir = true;
    }

    // 카메라 해상도 (Camera에서 가져오기)
    int w = -1, h = -1;
    auto it_cam = cameras.find(img.camera_id);
    if (it_cam != cameras.end()) {
      // 프로젝트에 맞는 쪽으로 한 줄만 쓰세요:
      // (1) 멤버 함수가 있을 때
      // w = static_cast<int>(it_cam->second.Width());
      // h = static_cast<int>(it_cam->second.Height());

      // (2) public 멤버 변수를 쓸 때
      w = static_cast<int>(it_cam->second.width);
      h = static_cast<int>(it_cam->second.height);
    }

    // 카메라 센터(초기 C = C_root + s*dir 로 가정)
    const Eigen::Vector3d C = c_root_fixed_ + s_val * dir;

    csv << img_id << ","
        << s_val << ","
        << (has_dir ? 1 : 0) << ","
        << dir.x() << "," << dir.y() << "," << dir.z() << ","
        << C.x()   << "," << C.y()   << "," << C.z()   << ","
        << w << "," << h << "\n";
  }
}

// Initialize s_i using MST where edge weights come from view_graph's match counts.
void GlobalPositioner::InitScalesByMST_FromViewGraphMatches(
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<track_t, Track>& tracks,
    image_t root_id) {

  // (0) Precondition: dir_param_holder_ is filled by BuildScaledCamDirectionsTree
  if (dir_param_holder_.empty()) {
    LOG(WARNING) << "[InitScalesByMST_FromViewGraphMatches] dir_param_holder_ is empty. "
                    "Call BuildScaledCamDirectionsTree first.";
  }

  // (1) Build adjacency and find max weight
  std::unordered_map<image_t, std::vector<std::pair<image_t,int>>> adj;
  int max_w = 0;
  for (const auto& [pair_id, P] : view_graph.image_pairs) {
    if (!P.is_valid) continue;
    const int w = static_cast<int>(P.inliers.size());
    if (w <= 0) continue;
    adj[P.image_id1].emplace_back(P.image_id2, w);
    adj[P.image_id2].emplace_back(P.image_id1, w);
    if (w > max_w) max_w = w;
  }
  if (max_w <= 0) max_w = 1;

  // (2) Build MST
  auto parent = BuildMST_MaxWeight(adj, images, root_id);

  // for (const auto& [pair_id, P] : view_graph.image_pairs) {
  //   if (!P.is_valid) continue;
  //   Eigen::Vector3d e_dir;
  //   if (GetEdgeWorldDirection(view_graph, images, P.image_id1, P.image_id2, &e_dir)) {
  //     auto& dv = dir_param_holder_[P.image_id2];
  //     if (dv.dot(e_dir) < 0) dv = -dv; // align sign
  //   }
  // }

  // (3) Make children list
  std::unordered_map<image_t, std::vector<image_t>> children;
  for (const auto& [v, p] : parent) {
    if (v == p) continue;
    children[p].push_back(v);
  }

  // (4) Prepare s arrays (stable addresses)
  s_vars_.clear();
  s_index_.clear();
  s_vars_.assign(images.size(), 0.0);
  {
    size_t idx = 0;
    for (const auto& [img_id, _] : images) s_index_[img_id] = idx++;
  }

  // (5) Root s = 0, BFS propagate
  s_vars_[s_index_.at(root_id)] = 0.0;
  std::queue<image_t> q;
  q.push(root_id);

  // NEW: counters for fallback statistics
  int nodes_considered = 0;
  int fallback_count   = 0;

  while (!q.empty()) {
    image_t u = q.front(); q.pop();
    for (image_t v : children[u]) {
      ++nodes_considered;
      // normalized weight
      int w_uv = 0;
      // find w_uv from adj list (could also keep a map<uint64_t,int>)
      if (adj.count(u)) {
        for (const auto& [to, wt] : adj[u]) if (to == v) { w_uv = wt; break; }
      }
      const double w_norm = static_cast<double>(w_uv) / static_cast<double>(max_w);

      // edge direction in world
      Eigen::Vector3d dir_edge_world(1,0,0);
      bool ok = GetEdgeWorldDirection(view_graph, images, u, v, &dir_edge_world);
      if (!ok) {
        // fallback to child's dir
        auto it_dir_v = dir_param_holder_.find(v);
        if (it_dir_v != dir_param_holder_.end()) dir_edge_world = it_dir_v->second;
      }

      // dot with child's direction
      auto it_dir_v = dir_param_holder_.find(v);
      const double dot_v = (it_dir_v != dir_param_holder_.end())
                           ? it_dir_v->second.dot(dir_edge_world)
                           : 1.0;

      const double s_u = s_vars_[s_index_.at(u)];

      // 2) fetch dirs
      // ... inside BFS over children[u]
      const auto& dir_u = dir_param_holder_.at(u);
      const auto& dir_v = dir_param_holder_.at(v);

      // 3) sampling-based refinement
      // Choose sampling hyper-parameters (tune as you like)
      auto rays_uv = CollectSharedWorldRays(u, v, images, tracks);
      const int R = static_cast<int>(rays_uv.size());

      // 30~60% accept
      const int    M_thresh    = std::clamp(static_cast<int>(0.4 * R), 8, 60);
      //0.9~1.1
      const double max_norm_xy = 2.5;
      const double min_z       = 1e-4;
      // Scale up the span
      const int    N_samples   = 151;      // was 101
      const double span        = 3.0;      // was 0.5
      const bool   both_cameras   = false;

      bool used_fallback = false;
      double s_v = PickSvByFrustum(u, v, s_u, dir_u, dir_v, c_root_fixed_,
                                  images, tracks,
                                  N_samples, span, M_thresh,
                                  max_norm_xy, min_z, both_cameras,
                                  &used_fallback);

      if (used_fallback) {
        VLOG(1) << "[InitScalesByMST] fallback on edge "
                << u << "->" << v
                << " R=" << R
                << " thresh=" << M_thresh
                << " s_u=" << s_u
                << " s_v=" << s_v;
      }

      // 4) assign
      s_vars_[s_index_.at(v)] = s_v;

      q.push(v);
    }
  }
  // NEW: print summary
  if (nodes_considered > 0) {
    const double ratio = 100.0 * static_cast<double>(fallback_count)
                                   / static_cast<double>(nodes_considered);
    LOG(INFO) << "[InitScalesByMST] Frustum voting fallback count = "
              << fallback_count << " / " << nodes_considered
              << " (" << ratio << "%)";
  } else {
    LOG(INFO) << "[InitScalesByMST] No child nodes considered (MST trivial).";
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
