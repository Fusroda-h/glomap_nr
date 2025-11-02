// glomap/estimators/global_positioning.cc

#include "glomap/estimators/global_positioning.h"
#include "glomap/estimators/cost_function.h"

#include <queue>
#include <numeric>
#include <algorithm>

namespace glomap {
namespace {

// Random 3D vector in [low, high].
Eigen::Vector3d RandVector3d(std::mt19937& random_generator,
                             double low,
                             double high) {
  std::uniform_real_distribution<double> distribution(low, high);
  return Eigen::Vector3d(distribution(random_generator),
                         distribution(random_generator),
                         distribution(random_generator));
}

// Undirected 64-bit edge key.
inline uint64_t MakeEdgeKey(image_t a, image_t b) {
  image_t u = std::min(a, b);
  image_t v = std::max(a, b);
  return (static_cast<uint64_t>(u) << 32) | static_cast<uint64_t>(v);
}

// Median of a vector.
inline double Median(std::vector<double> vals) {
  if (vals.empty()) return 0.0;
  const size_t mid = vals.size() / 2;
  std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
  return vals[mid];
}

}  // namespace

// ============================================================================
// 1) Triplet RANSAC for one triplet (your original C++)
//    (I kept your structure; only comments are English)
// ============================================================================

bool GlobalPositioner::EstimateSForTripletRansac(
    image_t i, image_t j, image_t k,
    const std::vector<track_t>& tids,
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<track_t, Track>& tracks,
    const std::unordered_map<camera_t, Camera>& cameras,
    Eigen::Vector3d* s_out) {
  // Fetch relative rotations and unit translations.
  Eigen::Matrix3d Rij, Rjk, Rik;
  Eigen::Vector3d t_hat_ij, t_hat_jk, t_hat_ik;

  if (!GetRelPose_I_to_J(view_graph, i, j, &Rij, &t_hat_ij)) return false;
  if (!GetRelPose_I_to_J(view_graph, j, k, &Rjk, &t_hat_jk)) return false;
  if (!GetRelPose_I_to_J(view_graph, i, k, &Rik, &t_hat_ik)) return false;

  // Build single-point 3x3 A.
  auto build_A = [&](track_t tid)->Eigen::Matrix3d {
    const int fi = FindObs(tracks.at(tid), i);
    const int fj = FindObs(tracks.at(tid), j);
    const int fk = FindObs(tracks.at(tid), k);
    Eigen::Vector3d di = NormalizeSafe(images.at(i).features_undist[fi]);
    Eigen::Vector3d B  = (Rjk * Rij - Rik) * di;

    Eigen::Matrix3d A;
    A.col(0) = B.cross(Rjk * t_hat_ij); // s_ij
    A.col(1) = B.cross(t_hat_jk);       // s_jk
    A.col(2) = B.cross(-t_hat_ik);      // s_ik
    return A;
  };

  auto getK = [&](image_t img_id, double* fx, double* fy, double* cx, double* cy){
    const camera_t cid = images.at(img_id).camera_id;
    const Camera&   cam = cameras.at(cid);
    GetIntrinsics(cam, fx, fy, cx, cy);
  };

  double fix,fiy,cix,ciy, fjx,fjy,cjx,cjy, fkx,fky,ckx,cky;
  getK(i, &fix,&fiy,&cix,&ciy);
  getK(j, &fjx,&fjy,&cjx,&cjy);
  getK(k, &fkx,&fky,&ckx,&cky);

  const Eigen::Matrix3d Eij = BuildEssential(Rij, t_hat_ij);
  const Eigen::Matrix3d Ejk = BuildEssential(Rjk, t_hat_jk);
  const Eigen::Matrix3d Eik = BuildEssential(Rik, t_hat_ik);

  const Eigen::Matrix3d Fij = EssentialToFundamental(Eij, fix,fiy,cix,ciy, fjx,fjy,cjx,cjy);
  const Eigen::Matrix3d Fjk = EssentialToFundamental(Ejk, fjx,fjy,cjx,cjy, fkx,fky,ckx,cky);
  const Eigen::Matrix3d Fik = EssentialToFundamental(Eik, fix,fiy,cix,ciy, fkx,fky,ckx,cky);

  const double ang_thr = Deg2Rad(options_.tri_inlier_ang_thresh_deg);
  auto angular_sum = [&](track_t tid, const Eigen::Vector3d& s, double* out_sum)->bool {
    const int fi = FindObs(tracks.at(tid), i);
    const int fj = FindObs(tracks.at(tid), j);
    const int fk = FindObs(tracks.at(tid), k);
    if (fi < 0 || fj < 0 || fk < 0) return false;

    const Eigen::Vector3d di = NormalizeSafe(images.at(i).features_undist[fi]);
    const Eigen::Vector3d dj = NormalizeSafe(images.at(j).features_undist[fj]);
    const Eigen::Vector3d dk = NormalizeSafe(images.at(k).features_undist[fk]);

    const Eigen::Vector3d tij = s(0) * t_hat_ij;
    const Eigen::Vector3d tjk = s(1) * t_hat_jk;

    Eigen::Matrix<double,6,3> A;
    Eigen::Matrix<double,6,1> b;
    A.block<3,1>(0,0) = -(Rij * di); A.block<3,1>(0,1) = dj; A.block<3,1>(0,2) = Eigen::Vector3d::Zero();
    b.segment<3>(0)    =  tij;
    A.block<3,1>(3,0) =  Eigen::Vector3d::Zero(); A.block<3,1>(3,1) = -(Rjk * dj); A.block<3,1>(3,2) = dk;
    b.segment<3>(3)    =  tjk;

    const Eigen::Vector3d lambda = A.colPivHouseholderQr().solve(b);
    if (!lambda.allFinite()) return false;
    if (lambda(0)<=options_.tri_min_depth || lambda(1)<=options_.tri_min_depth || lambda(2)<=options_.tri_min_depth) return false;

    const Eigen::Vector3d Xi = lambda(0)*di;
    const Eigen::Vector3d Xj = Rij*Xi + tij;
    const Eigen::Vector3d Xk = Rjk*Xj + tjk;

    const double e_i = AngleRad(Xi, di);
    const double e_j = AngleRad(Xj, dj);
    const double e_k = AngleRad(Xk, dk);
    *out_sum = e_i + e_j + e_k;
    return std::isfinite(*out_sum);
  };

  const double px_thr = options_.tri_inlier_px_thresh;
  const double px_thr_sum_sq = 3.0 * (px_thr * px_thr);
  auto pixel_sum = [&](track_t tid, const Eigen::Vector3d& s, double* out_sum)->bool {
    const int fi = FindObs(tracks.at(tid), i);
    const int fj = FindObs(tracks.at(tid), j);
    const int fk = FindObs(tracks.at(tid), k);
    if (fi < 0 || fj < 0 || fk < 0) return false;

    const Eigen::Vector3d di = NormalizeSafe(images.at(i).features_undist[fi]);
    const Eigen::Vector3d dj = NormalizeSafe(images.at(j).features_undist[fj]);
    const Eigen::Vector3d dk = NormalizeSafe(images.at(k).features_undist[fk]);

    const Eigen::Vector3d xi = RayToPixH(di, fix,fiy,cix,ciy);
    const Eigen::Vector3d xj = RayToPixH(dj, fjx,fjy,cjx,cjy);
    const Eigen::Vector3d xk = RayToPixH(dk, fkx,fky,ckx,cky);

    const Eigen::Vector3d tij = s(0) * t_hat_ij;
    const Eigen::Vector3d tjk = s(1) * t_hat_jk;

    Eigen::Matrix<double,6,3> A;
    Eigen::Matrix<double,6,1> b;
    A.block<3,1>(0,0) = -(Rij * di); A.block<3,1>(0,1) = dj; A.block<3,1>(0,2) = Eigen::Vector3d::Zero();
    b.segment<3>(0)    =  tij;
    A.block<3,1>(3,0) =  Eigen::Vector3d::Zero(); A.block<3,1>(3,1) = -(Rjk * dj); A.block<3,1>(3,2) = dk;
    b.segment<3>(3)    =  tjk;

    const Eigen::Vector3d lambda = A.colPivHouseholderQr().solve(b);
    if (!lambda.allFinite()) return false;
    if (lambda(0)<=options_.tri_min_depth || lambda(1)<=options_.tri_min_depth || lambda(2)<=options_.tri_min_depth) return false;

    const Eigen::Vector3d Xi = lambda(0)*di;
    const Eigen::Vector3d Xj = Rij*Xi + tij;
    const Eigen::Vector3d Xk = Rjk*Xj + tjk;

    const Eigen::Vector3d xj_pred = RayToPixH(Xj, fjx,fjy,cjx,cjy);
    const Eigen::Vector3d xk_pred = RayToPixH(Xk, fkx,fky,ckx,cky);
    const Eigen::Vector3d xi_pred = RayToPixH(Xi, fix,fiy,cix,ciy);

    const double e_ij = (xj.head<2>() - xj_pred.head<2>()).squaredNorm();
    const double e_jk = (xk.head<2>() - xk_pred.head<2>()).squaredNorm();
    const double e_ik = (xi.head<2>() - xi_pred.head<2>()).squaredNorm();
    *out_sum = e_ij + e_jk + e_ik;
    return std::isfinite(*out_sum);
  };

  const double sam_thr_sum_sq = 3.0 * (px_thr * px_thr);
  auto sampson_sum = [&](track_t tid, const Eigen::Vector3d& /*s*/, double* out_sum)->bool {
    const int fi = FindObs(tracks.at(tid), i);
    const int fj = FindObs(tracks.at(tid), j);
    const int fk = FindObs(tracks.at(tid), k);
    if (fi < 0 || fj < 0 || fk < 0) return false;

    const Eigen::Vector3d di = NormalizeSafe(images.at(i).features_undist[fi]);
    const Eigen::Vector3d dj = NormalizeSafe(images.at(j).features_undist[fj]);
    const Eigen::Vector3d dk = NormalizeSafe(images.at(k).features_undist[fk]);

    const Eigen::Vector3d xi = RayToPixH(di, fix,fiy,cix,ciy);
    const Eigen::Vector3d xj = RayToPixH(dj, fjx,fjy,cjx,cjy);
    const Eigen::Vector3d xk = RayToPixH(dk, fkx,fky,ckx,cky);

    const double e_ij = SampsonErrorSq(Fij, xj, xi);
    const double e_jk = SampsonErrorSq(Fjk, xk, xj);
    const double e_ik = SampsonErrorSq(Fik, xk, xi);

    *out_sum = e_ij + e_jk + e_ik;
    return std::isfinite(*out_sum);
  };

  std::function<bool(track_t, const Eigen::Vector3d&, double*)> consensus_fn;
  double inlier_thr_sum = 0.0;

  switch (options_.tri_consensus_metric) {
    case GlobalPositionerOptions::TriConsensusMetric::ANGULAR:
      consensus_fn = angular_sum;
      inlier_thr_sum = 3.0 * ang_thr;
      break;
    case GlobalPositionerOptions::TriConsensusMetric::PIXEL_REPROJ:
      consensus_fn = pixel_sum;
      inlier_thr_sum = px_thr_sum_sq;
      break;
    case GlobalPositionerOptions::TriConsensusMetric::SAMPSON:
      consensus_fn = sampson_sum;
      inlier_thr_sum = sam_thr_sum_sq;
      break;
  }

  std::mt19937 rng(options_.seed);
  if (tids.empty()) return false;
  std::uniform_int_distribution<int> uni(0, static_cast<int>(tids.size()) - 1);

  int best_inl = -1;
  Eigen::Vector3d best_s(1,0,0);

  for (int it = 0; it < options_.tri_ransac_max_iters; ++it) {
    const track_t t0 = tids[uni(rng)];

    Eigen::Matrix3d Am = build_A(t0);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Am, Eigen::ComputeFullV);
    Eigen::Vector3d s = svd.matrixV().col(2);
    if (!s.allFinite() || s.norm() < 1e-12) continue;
    s.normalize();

    if (s(1) < 0) s = -s;

    int inl = 0;
    for (auto tid : tids) {
      double err_sum = 0.0;
      if (!consensus_fn(tid, s, &err_sum)) continue;
      if (err_sum < inlier_thr_sum) ++inl;
    }

    if (inl > best_inl) {
      best_inl = inl;
      best_s   = s;
    }
  }
  if (best_inl < options_.tri_min_inliers) return false;

  std::vector<Eigen::Matrix3d> As; As.reserve(best_inl);
  for (auto tid : tids) {
    double err_sum = 0.0;
    if (!consensus_fn(tid, best_s, &err_sum)) continue;
    if (err_sum < inlier_thr_sum) {
      As.emplace_back(build_A(tid));
    }
  }
  if (As.empty()) return false;

  Eigen::MatrixXd Astack(3 * static_cast<int>(As.size()), 3);
  for (int r = 0; r < static_cast<int>(As.size()); ++r) {
    Astack.block<3,3>(3*r, 0) = As[r];
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(Astack, Eigen::ComputeThinV);
  Eigen::Vector3d s = svd.matrixV().col(2);
  if (s(1) < 0) s = -s;
  *s_out = s / s.norm();
  return true;
}

// ============================================================================
// 2) Collect edge scales for all triplets (your original C++)
// ============================================================================

void GlobalPositioner::EstimateEdgeScalesByTriRansac(
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<track_t, Track>& tracks,
    const std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<uint64_t, std::vector<double>>& edge_scales) {

  struct Obs { image_t img; int feat_idx; };
  std::unordered_map<track_t, std::vector<Obs>> track_obs;
  track_obs.reserve(tracks.size());

  for (const auto& [tid, tr] : tracks) {
    std::vector<Obs> obs_list;
    obs_list.reserve(tr.observations.size());
    for (const auto& ob : tr.observations) {
      auto it = images.find(ob.first);
      if (it == images.end()) continue;
      if (!it->second.is_registered) continue;
      obs_list.push_back({ob.first, static_cast<int>(ob.second)});
    }
    if (static_cast<int>(obs_list.size()) >= 3) {
      track_obs[tid] = std::move(obs_list);
    }
  }

  struct TriKey {
    image_t i, j, k;
    bool operator==(const TriKey& o) const { return i==o.i && j==o.j && k==o.k; }
  };
  struct TriKeyHash {
    size_t operator()(const TriKey& t) const {
      return (size_t)t.i*1315423911u ^ (size_t)t.j*2654435761u ^ (size_t)t.k;
    }
  };

  auto make_tri_key = [](image_t a, image_t b, image_t c){
    if (a>b) std::swap(a,b);
    if (b>c) std::swap(b,c);
    if (a>b) std::swap(a,b);
    return TriKey{a,b,c};
  };

  std::unordered_map<TriKey, std::vector<track_t>, TriKeyHash> tri_points;

  for (const auto& [tid, obs] : track_obs) {
    const int T = static_cast<int>(obs.size());
    int emitted = 0;
    for (int a=0; a<T && emitted<options_.tri_max_triplets_per_track; ++a)
      for (int b=a+1; b<T && emitted<options_.tri_max_triplets_per_track; ++b)
        for (int c=b+1; c<T && emitted<options_.tri_max_triplets_per_track; ++c) {
          TriKey key = make_tri_key(obs[a].img, obs[b].img, obs[c].img);
          tri_points[key].push_back(tid);
          ++emitted;
        }
  }

  auto push_edge = [&](image_t a, image_t b, double s_ab){
    const uint64_t k = MakeEdgeKey(a, b);
    edge_scales[k].push_back(std::abs(s_ab));
  };

  int accepted = 0;
  for (const auto& [key, tids3] : tri_points) {
    if (static_cast<int>(tids3.size()) < options_.tri_min_inliers) continue;

    Eigen::Vector3d s_hat;
    if (!EstimateSForTripletRansac(key.i, key.j, key.k, tids3,
                                   view_graph, images, tracks, cameras, &s_hat)) {
      continue;
    }
    ++accepted;

    push_edge(key.i, key.j, s_hat(0));
    push_edge(key.j, key.k, s_hat(1));
    push_edge(key.i, key.k, s_hat(2));
  }

  LOG(INFO) << "[TRI_RANSAC/" << TriMetricTag(options_) << "] "
            << "accepted triplets: " << accepted << " / " << tri_points.size()
            << ", edges with samples: " << edge_scales.size()
            << ", iters=" << options_.tri_ransac_max_iters
            << ", min_inliers=" << options_.tri_min_inliers
            << ", max_triplets_per_track=" << options_.tri_max_triplets_per_track;
}

// ============================================================================
// ctor
// ============================================================================
GlobalPositioner::GlobalPositioner(const GlobalPositionerOptions& options)
    : options_(options) {
  random_generator_.seed(options_.seed);
}

// ============================================================================
// Solve
// ============================================================================
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

  // SPECIAL PATH for ONLY_POINTS: use triplet-based scale initialization.
  if (options_.constraint_type == GlobalPositionerOptions::ONLY_POINTS) {
    LOG(INFO) << "[GlobalPositioner] ONLY_POINTS: using triplet-based init.";

    std::unordered_map<uint64_t, std::vector<double>> edge_scales;
    EstimateEdgeScalesByTriRansac(view_graph, images, tracks, cameras,
                                  edge_scales);

    if (!edge_scales.empty()) {
      InitializeCamerasFromTriScales(view_graph, images, edge_scales);
      InitializePointsFromCameras(cameras, images, tracks);
      ConvertResults(images);
      return true;
    } else {
      LOG(WARNING) << "[GlobalPositioner/ONLY_POINTS] no edge scales from "
                      "triplet RANSAC; falling back to legacy pipeline.";
      // fall-through to legacy code below
    }
  }

  LOG(INFO) << "Setting up the global positioner problem";

  SetupProblem(view_graph, tracks);

  // NOTE: original method name is InitializeRandomPositions(...)
  InitializeRandomPositions(view_graph, images, tracks);

  if (options_.constraint_type != GlobalPositionerOptions::ONLY_POINTS) {
    AddCameraToCameraConstraints(view_graph, images);
  }
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

// ============================================================================
// NEW: place cameras from triplet edge scales
// ============================================================================
void GlobalPositioner::InitializeCamerasFromTriScales(
    const ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    const std::unordered_map<uint64_t, std::vector<double>>& edge_scales) {
  if (images.empty()) return;

  // Build adjacency from view-graph
  std::unordered_map<image_t, std::vector<image_t>> adj;
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;
    const image_t i = image_pair.image_id1;
    const image_t j = image_pair.image_id2;
    auto it_i = images.find(i);
    auto it_j = images.find(j);
    if (it_i == images.end() || it_j == images.end()) continue;
    if (!it_i->second.is_registered || !it_j->second.is_registered) continue;
    adj[i].push_back(j);
    adj[j].push_back(i);
  }
  if (adj.empty()) {
    LOG(WARNING) << "[TriInit] adjacency is empty; cannot initialize cameras.";
    return;
  }

  // choose a root
  const image_t root = images.begin()->first;
  images[root].cam_from_world.translation = Eigen::Vector3d::Zero();

  std::queue<image_t> q;
  std::unordered_set<image_t> visited;
  q.push(root);
  visited.insert(root);

  int placed = 1;

  while (!q.empty()) {
    const image_t u = q.front(); q.pop();
    const Eigen::Vector3d Cu = images[u].cam_from_world.translation;
    auto it_adj = adj.find(u);
    if (it_adj == adj.end()) continue;

    for (const image_t v : it_adj->second) {
      if (visited.count(v)) continue;

      const uint64_t key = MakeEdgeKey(u, v);
      auto it_s = edge_scales.find(key);
      if (it_s == edge_scales.end() || it_s->second.empty()) {
        LOG(WARNING) << "[TriInit] missing scale for edge (" << u << "," << v
                     << "), skipping.";
        continue;
      }
      const double s_uv = Median(it_s->second);

      Eigen::Vector3d dir_world;
      if (!GetWorldDirection(view_graph, images, u, v, &dir_world)) {
        LOG(WARNING) << "[TriInit] cannot get direction for edge (" << u
                     << "," << v << "), skipping.";
        continue;
      }

      images[v].cam_from_world.translation = Cu + s_uv * dir_world;
      visited.insert(v);
      q.push(v);
      ++placed;
    }
  }

  LOG(INFO) << "[TriInit] initialized " << placed << " camera centers.";
}

// ============================================================================
// NEW: get world direction u -> v from view-graph
// (use quaternions exactly like original GLOMAP code)
// ============================================================================
bool GlobalPositioner::GetWorldDirection(
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    image_t src,
    image_t dst,
    Eigen::Vector3d* dir_world) const {
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;

    const image_t id1 = image_pair.image_id1;
    const image_t id2 = image_pair.image_id2;

    // src -> dst
    if (id1 == src && id2 == dst) {
      auto it_dst = images.find(dst);
      if (it_dst == images.end()) return false;
      const Eigen::Quaterniond& q_wc_dst =
          it_dst->second.cam_from_world.rotation;
      // same pattern as in AddCameraToCameraConstraints()
      Eigen::Vector3d d =
          -(q_wc_dst.inverse() * image_pair.cam2_from_cam1.translation);
      const double n = d.norm();
      if (n < 1e-12) return false;
      *dir_world = d / n;
      return true;
    }

    // dst -> src (reverse direction)
    if (id1 == dst && id2 == src) {
      auto it_src = images.find(src);
      if (it_src == images.end()) return false;
      const Eigen::Quaterniond& q_wc_src =
          it_src->second.cam_from_world.rotation;
      Eigen::Vector3d d =
          -(q_wc_src.inverse() * image_pair.cam2_from_cam1.translation);
      const double n = d.norm();
      if (n < 1e-12) return false;
      *dir_world = -d / n;
      return true;
    }
  }
  return false;
}

// ============================================================================
// NEW: triangulate tracks from the initialized cameras
// ============================================================================
void GlobalPositioner::InitializePointsFromCameras(
    std::unordered_map<camera_t, Camera>& /*cameras*/,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {

  int tri_ok = 0, tri_fail = 0;

  for (auto& [track_id, track] : tracks) {
    if (track.observations.size() < options_.min_num_view_per_track) {
      ++tri_fail;
      continue;
    }

    Eigen::Matrix3d ATA = Eigen::Matrix3d::Zero();
    Eigen::Vector3d ATb = Eigen::Vector3d::Zero();
    int used = 0;

    for (const auto& ob : track.observations) {
      const image_t img_id = ob.first;
      const int feat_idx = static_cast<int>(ob.second);

      auto it_img = images.find(img_id);
      if (it_img == images.end()) continue;
      const Image& img = it_img->second;
      if (!img.is_registered) continue;

      // camera center in world
      const Eigen::Vector3d& C = img.cam_from_world.translation;
      // bearing in camera frame
      const Eigen::Vector3d& f_cam = img.features_undist[feat_idx];
      // rotate to world via quaternion
      Eigen::Vector3d d_world = img.cam_from_world.rotation.inverse() * f_cam;
      d_world.normalize();

      const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
      const Eigen::Matrix3d P = I - d_world * d_world.transpose();

      ATA += P;
      ATb += P * C;
      ++used;
    }

    if (used < 2) {
      ++tri_fail;
      continue;
    }

    Eigen::Vector3d X = ATA.ldlt().solve(ATb);
    if (!X.allFinite()) {
      ++tri_fail;
      continue;
    }

    track.xyz = X;
    track.is_initialized = true;
    ++tri_ok;
  }

  LOG(INFO) << "[TriInit] triangulated tracks ok=" << tri_ok
            << " fail=" << tri_fail;
}

// ============================================================================
// Original GLOMAP parts (unchanged)
// ============================================================================

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
    if (image_pair.is_valid == false) continue;

    constrained_positions.insert(image_pair.image_id1);
    constrained_positions.insert(image_pair.image_id2);
  }

  if (options_.constraint_type != GlobalPositionerOptions::ONLY_CAMERAS) {
    for (const auto& [track_id, track] : tracks) {
      if (track.observations.size() < options_.min_num_view_per_track) continue;
      for (const auto& observation : track.observations) {
        auto it = images.find(observation.first);
        if (it == images.end()) continue;
        if (!it->second.is_registered) continue;
        constrained_positions.insert(observation.first);
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
        -(image.cam_from_world.rotation * image.cam_from_world.translation);
  }
}

}  // namespace glomap
