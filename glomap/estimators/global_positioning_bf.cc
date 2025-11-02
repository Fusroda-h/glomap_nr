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
#include <Eigen/SVD>
#include <unordered_set>
#include <random>
#include <functional>

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

// Solve A*[s, alpha] = b  where rows are [x_k, u] and y_k = s*x_k + alpha*u
static bool SolveSAlphaLsq(
    const std::vector<Eigen::Vector3d>& Xi_cam_i,   // points in cam-i
    const std::vector<Eigen::Vector3d>& Xj_cam_j,   // points in cam-j
    const Eigen::Matrix3d& R_ij,                    // cam-j from cam-i
    const Eigen::Vector3d& uhat_j,                  // unit t in cam-j
    double eps, int min_pts,
    double* s, double* alpha, double* rms)
{
  const int N = (int)std::min(Xi_cam_i.size(), Xj_cam_j.size());
  if (N < min_pts) return false;

  Eigen::MatrixXd A(3*N, 2);
  Eigen::VectorXd b(3*N);

  for (int k=0; k<N; ++k) {
    // Python: x = Xi @ R.T  (Xi row-vectors) ⇒ here: x = R^T * Xi (col)
    const Eigen::Vector3d x = R_ij.transpose() * Xi_cam_i[k];
    const Eigen::Vector3d y = Xj_cam_j[k];

    A.block<3,1>(3*k, 0) = x;
    A.block<3,1>(3*k, 1) = uhat_j;
    b.segment<3>(3*k)    = y;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto& S = svd.singularValues();
  Eigen::VectorXd Sinv = S.unaryExpr([&](double sv){ return (sv > eps) ? 1.0/sv : 0.0; });

  // beta = V * S^+ * U^T * b
  const Eigen::VectorXd beta = svd.matrixV() * Sinv.asDiagonal() * (svd.matrixU().transpose() * b);

  if (s)     *s     = beta(0);
  if (alpha) *alpha = beta(1);

  if (rms) {
    Eigen::VectorXd ypred = A * beta;
    *rms = std::sqrt((b - ypred).squaredNorm() / (3.0*N));
  }
  return true;
}

// ========================= NEW UTILS FOR TRIPLET RANSAC =========================

// // Return obs index of an image in a track, or -1 if not observed.
static inline int FindObs(const glomap::Track& tr, image_t img_id) {
  for (const auto& ob : tr.observations) {
    if (ob.first == img_id) return static_cast<int>(ob.second);
  }
  return -1;
}

// Normalize a 3D vector with epsilon guard.
static inline Eigen::Vector3d NormalizeSafe(const Eigen::Vector3d& v, double eps=1e-12) {
  const double n = v.norm();
  if (n <= eps) return v;
  return v / n;
}

// Angle between two 3D vectors (radians), guard against NaN.
static inline double AngleRad(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  const double c = std::clamp(NormalizeSafe(a).dot(NormalizeSafe(b)), -1.0, 1.0);
  return std::acos(c);
}

static inline double Deg2Rad(double d) { return d * M_PI / 180.0; }

// Fetch relative pose (i -> j): Rij (cam-j from cam-i), t_hat_j (unit translation in cam-j frame).
// Returns false if not found/degenerate.
static bool GetRelPose_I_to_J(
    const ViewGraph& view_graph,
    image_t i, image_t j,
    Eigen::Matrix3d* Rij /*out*/,
    Eigen::Vector3d* t_hat_j /*out*/) {

  for (const auto& [pair_id, P] : view_graph.image_pairs) {
    if (!P.is_valid) continue;
    if (P.image_id1 == i && P.image_id2 == j) {
      const Eigen::Quaterniond q = P.cam2_from_cam1.rotation;
      const Eigen::Vector3d   t = P.cam2_from_cam1.translation; // in cam-j frame
      *Rij = q.toRotationMatrix();

      const double n = t.norm();
      if (n <= 1e-12) return false;
      *t_hat_j = t / n;
      return true;
    }
    if (P.image_id1 == j && P.image_id2 == i) {
      // We need pose from i->j; current is j->i, so invert.
      const Eigen::Quaterniond qji = P.cam2_from_cam1.rotation;
      const Eigen::Vector3d   tji  = P.cam2_from_cam1.translation; // in cam-i frame

      // Inverse: R_ij = R_ji^T ;  t_ij (in cam-j) = -R_ji * t_ji
      const Eigen::Matrix3d Rji = qji.toRotationMatrix();
      *Rij = Rji.transpose();
      const Eigen::Vector3d tij_camj = -Rji * tji;
      const double n = tij_camj.norm();
      if (n <= 1e-12) return false;
      *t_hat_j = tij_camj / n;
      return true;
    }
  }
  return false;
}

// =================== intrinsics helpers (consistent with ViewGraphCalibrator) ===================
// Use isotropic intrinsics (fx=fy) as the pipeline copies a single scalar focal to both indices.
static inline void GetIntrinsics(const Camera& cam, double* fx, double* fy, double* cx, double* cy) {
  const double f = cam.Focal();
  const Eigen::Vector2d pp = cam.PrincipalPoint();
  *fx = f; *fy = f; *cx = pp.x(); *cy = pp.y();
}

// Project a normalized cam-ray to pixel homogeneous using (fx, fy, cx, cy)
static inline Eigen::Vector3d RayToPixH(const Eigen::Vector3d& d, double fx, double fy, double cx, double cy) {
  const double z = std::max(d.z(), 1e-12);
  const double u = fx * (d.x()/z) + cx;
  const double v = fy * (d.y()/z) + cy;
  return Eigen::Vector3d(u, v, 1.0);
}

// Build skew matrix [t]_x
static inline Eigen::Matrix3d Skew(const Eigen::Vector3d& t) {
  Eigen::Matrix3d T;
  T <<     0, -t.z(),  t.y(),
        t.z(),     0, -t.x(),
       -t.y(),  t.x(),     0;
  return T;
}

// Essential and Fundamental (unit translation is fine for Sampson)
static inline Eigen::Matrix3d BuildEssential(const Eigen::Matrix3d& Rij, const Eigen::Vector3d& t_hat_j) {
  return Skew(t_hat_j) * Rij;
}
static inline Eigen::Matrix3d Kinv(double fx, double fy, double cx, double cy) {
  Eigen::Matrix3d Ki;
  Ki << 1.0/fx,     0.0,   -cx/fx,
           0.0,  1.0/fy,   -cy/fy,
           0.0,     0.0,      1.0;
  return Ki;
}
static inline Eigen::Matrix3d EssentialToFundamental(const Eigen::Matrix3d& E,
                                                     double fix, double fiy, double cix, double ciy,
                                                     double fjx, double fjy, double cjx, double cjy) {
  const Eigen::Matrix3d Ki_inv  = Kinv(fix, fiy, cix, ciy);
  const Eigen::Matrix3d Kj_invT = Kinv(fjx, fjy, cjx, cjy).transpose();
  return Kj_invT * E * Ki_inv;
}

// Sampson error (squared pixels approx)
static inline double SampsonErrorSq(const Eigen::Matrix3d& F,
                                    const Eigen::Vector3d& xj, const Eigen::Vector3d& xi) {
  const Eigen::Vector3d Fx  = F * xi;
  const Eigen::Vector3d FTx = F.transpose() * xj;
  const double num = xj.transpose() * F * xi;
  const double den = Fx.x()*Fx.x() + Fx.y()*Fx.y() + FTx.x()*FTx.x() + FTx.y()*FTx.y();
  return (num*num) / std::max(den, 1e-12);
}

// consensus metric name + threshold 요약
static inline std::string TriMetricTag(
    const GlobalPositionerOptions& opt) {
  using M = GlobalPositionerOptions::TriConsensusMetric;
  switch (opt.tri_consensus_metric) {
    case M::ANGULAR: {
      std::ostringstream os;
      os << "ANGULAR(" << opt.tri_inlier_ang_thresh_deg << "deg)";
      return os.str();
    }
    case M::PIXEL_REPROJ: {
      std::ostringstream os;
      os << "PIXEL(" << opt.tri_inlier_px_thresh << "px)";
      return os.str();
    }
    case M::SAMPSON: {
      std::ostringstream os;
      os << "SAMPSON(" << opt.tri_inlier_px_thresh << "px)";
      return os.str();
    }
  }
  return "UNKNOWN";
}

// Robust (s, alpha) estimation on a single MST edge (u -> v) using RANSAC.
// Xi: points expressed in camera-i coordinates
// Xj: points expressed in camera-j coordinates
// R_ij: rotation from camera-i to camera-j (cam-j from cam-i)
// uhat_j: unit translation direction in camera-j frame (t_hat_ij)
// Returns true if successful and writes s_out, a_out, rms_out.

// ---- 본체 (픽셀 합의까지 지원) ----
static bool EdgeRansacSolveSAlpha(
    const std::vector<Eigen::Vector3d>& Xi,
    const std::vector<Eigen::Vector3d>& Xj,
    const Eigen::Matrix3d& R_ij,
    const Eigen::Vector3d& uhat_j,
    const glomap::GlobalPositionerOptions& opt,
    const std::vector<Eigen::Vector2d>* xj_px, // may be null for ANGULAR
    double fx_j, double fy_j, double cx_j, double cy_j,
    double* s_out, double* a_out, double* rms_out)
{
  struct SAResult { double s=0, alpha=0, rms=1e9; int inliers=-1; };

  const int N = (int)std::min({Xi.size(), Xj.size(), xj_px ? xj_px->size() : Xj.size()});
  if (N < opt.s_alpha_min_pts) return false;

  // RANSAC hyper-parameters (sensible defaults)
  const int    max_iters   = 5000;
  const int    min_inliers = std::max(3, opt.s_alpha_min_pts);
  const double eps         = std::max(1e-12, opt.s_alpha_eps);

  // Which consensus metric to use for this edge?
  const bool use_px =
      (opt.edge_consensus_metric ==
      glomap::GlobalPositionerOptions::EdgeConsensusMetric::PIXEL_REPROJ);

  // Thresholds (per-edge)
  const double thr_ang =
      std::max(1e-6, opt.s_alpha_inlier_ang_thresh_deg * M_PI / 180.0);
  const double thr_px =
      std::max(1e-6, opt.s_alpha_px_thresh);


  std::mt19937 rng(opt.seed);
  std::uniform_int_distribution<int> uni(0, N-1);

  auto solve_lsq = [&](const std::vector<int>& idxs, SAResult* res)->bool {
    std::vector<Eigen::Vector3d> Xi_s, Xj_s;
    Xi_s.reserve(idxs.size()); Xj_s.reserve(idxs.size());
    for (int id : idxs) { Xi_s.push_back(Xi[id]); Xj_s.push_back(Xj[id]); }
    double s,a,rms;
    if (!SolveSAlphaLsq(Xi_s, Xj_s, R_ij, uhat_j, eps,
                        (int)idxs.size(), &s,&a,&rms)) return false;
    res->s=s; res->alpha=a; res->rms=rms; res->inliers=0;
    return std::isfinite(s) && std::isfinite(a);
  };

  auto angular_err = [&](int k, const SAResult& m)->double {
    const Eigen::Vector3d xj_from_i = R_ij.transpose() * Xi[k];
    const Eigen::Vector3d y_pred    = m.s * xj_from_i + m.alpha * uhat_j;
    const double c = std::clamp(y_pred.normalized().dot(Xj[k].normalized()), -1.0, 1.0);
    return std::acos(c);
  };

  auto reproj_px_err = [&](int k, const SAResult& m)->double {
    if (!xj_px) return 1e12;
    const Eigen::Vector2d& x_obs = (*xj_px)[k];
    const Eigen::Vector3d xj_from_i = R_ij.transpose() * Xi[k];
    Eigen::Vector3d y = m.s * xj_from_i + m.alpha * uhat_j;
    if (y.z() <= 1e-12) y.z() = 1e-12;
    const double u = fx_j * (y.x()/y.z()) + cx_j;
    const double v = fy_j * (y.y()/y.z()) + cy_j;
    const double du = u - x_obs.x(), dv = v - x_obs.y();
    return std::sqrt(du*du + dv*dv);
  };

  SAResult best;

  for (int it=0; it<max_iters; ++it) {
    std::vector<int> idxs(1); idxs[0] = uni(rng);
    SAResult m;
    if (!solve_lsq(idxs, &m)) continue;

    int inl = 0; double acc = 0.0;
    for (int k=0;k<N;++k) {
      double e = use_px ? reproj_px_err(k,m) : angular_err(k,m);
      if (!std::isfinite(e)) continue;
      const bool ok = use_px ? (e < thr_px) : (e < thr_ang);
      if (ok) { ++inl; acc += (use_px ? e*e : e*e); }
    }
    if (inl > best.inliers) {
      best = m;
      best.inliers = inl;
      best.rms = (inl>0) ? std::sqrt(acc / inl) : 1e9;
    }
  }

  if (best.inliers < min_inliers) {
    if (VLOG_IS_ON(1)) {
      // 샘플 200개만 훑어서 에러 통계
      const int M = std::min(N, 200);
      std::vector<double> errs; errs.reserve(M);
      SAResult mdbg; mdbg.s=0; mdbg.alpha=0; // trivial
      for (int k=0;k<M;++k) {
        double e = use_px ? reproj_px_err(k, mdbg) : angular_err(k, mdbg);
        if (std::isfinite(e)) errs.push_back(e);
      }
      if (!errs.empty()) {
        std::nth_element(errs.begin(), errs.begin()+errs.size()/2, errs.end());
        double med = errs[errs.size()/2];
        double mn = *std::min_element(errs.begin(), errs.end());
        double mx = *std::max_element(errs.begin(), errs.end());
        LOG(INFO) << "[S_ALPHA/RANSAC] no-inlier: N="<<N
                  << " use_px="<<use_px
                  << " thr="<<(use_px?thr_px:thr_ang)
                  << " err[min/med/max]="<<mn<<"/"<<med<<"/"<<mx;
      } else {
        LOG(INFO) << "[S_ALPHA/RANSAC] no-inlier: N="<<N
                  << " use_px="<<use_px
                  << " thr="<<(use_px?thr_px:thr_ang)
                  << " (no finite errs)";
      }
    }
    return false;
  }

  // re-fit with inliers
  std::vector<Eigen::Vector3d> Xi_inl, Xj_inl;
  Xi_inl.reserve(best.inliers); Xj_inl.reserve(best.inliers);
  for (int k=0;k<N;++k) {
    double e = use_px ? reproj_px_err(k,best) : angular_err(k,best);
    const bool ok = std::isfinite(e) && (use_px ? (e < thr_px) : (e < thr_ang));
    if (ok) { Xi_inl.push_back(Xi[k]); Xj_inl.push_back(Xj[k]); }
  }

  double s,a,rms;
  if (!SolveSAlphaLsq(Xi_inl, Xj_inl, R_ij, uhat_j, eps, (int)Xi_inl.size(), &s,&a,&rms))
    return false;

  if (s_out) *s_out = s;
  if (a_out) *a_out = a;
  if (rms_out) *rms_out = best.rms; // 또는 rms 사용 가능
  return true;
}

struct SharedRayWithPixel {
  Eigen::Vector3d ru_world;   // u의 월드 방향
  Eigen::Vector3d rv_world;   // v의 월드 방향
  Eigen::Vector2d xj_px;      // v에서의 관측 픽셀 (fx,fy,cx,cy로 fv를 투영)
};

// u,v가 공통으로 보는 트랙에 대해 (ru, rv, xj_px)를 모은다.
static std::vector<SharedRayWithPixel>
CollectSharedRaysWithPixels(image_t u, image_t v,
                            const std::unordered_map<image_t, Image>& images,
                            const std::unordered_map<track_t, Track>& tracks,
                            const std::unordered_map<camera_t, Camera>& cameras) {
  std::vector<SharedRayWithPixel> out;
  out.reserve(256);

  const auto& img_u = images.at(u);
  const auto& img_v = images.at(v);
  if (!img_u.is_registered || !img_v.is_registered) return out;

  double fx, fy, cx, cy;
  {
    const camera_t cid_v = img_v.camera_id;
    const Camera& cam_v = cameras.at(cid_v);
    GetIntrinsics(cam_v, &fx, &fy, &cx, &cy);
  }

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

    // v의 픽셀 (fv를 바로 투영)
    const double z = std::max(fv.z(), 1e-12);
    const double u_px = fx * (fv.x() / z) + cx;
    const double v_px = fy * (fv.y() / z) + cy;

    out.push_back({ru, rv, Eigen::Vector2d(u_px, v_px)});
  }
  return out;
}

// somewhere in global_positioning.cc (또는 .h/.cc 유틸로)
static bool LoadCentersCSV(const std::string& path,
                           std::unordered_map<image_t, Eigen::Vector3d>* out) {
  out->clear();
  std::ifstream in(path);
  if (!in.is_open()) {
    LOG(ERROR) << "Failed to open CSV: " << path;
    return false;
  }
  std::string line;
  bool header_checked = false;
  size_t n_ok = 0, n_bad = 0;

  while (std::getline(in, line)) {
    // trim
    if (line.empty()) continue;
    // 헤더 감지: 알파벳으로 시작하거나 'image_id' 포함
    if (!header_checked) {
      header_checked = true;
      std::string low = line;
      std::transform(low.begin(), low.end(), low.begin(), ::tolower);
      if (low.find("image_id") != std::string::npos) continue; // skip header
    }

    // 토큰 분리 (콤마 또는 스페이스)
    for (char& c : line) if (c == ',') c = ' ';
    std::istringstream iss(line);
    long long iid; double cx, cy, cz;
    if (!(iss >> iid >> cx >> cy >> cz)) {
      ++n_bad; continue;
    }
    (*out)[static_cast<image_t>(iid)] = Eigen::Vector3d(cx, cy, cz);
    ++n_ok;
  }
  LOG(INFO) << "[LoadCentersCSV] loaded " << n_ok << " centers, skipped " << n_bad
            << " lines from " << path;
  return n_ok > 0;
}

// 트랙을 이미 초기화된 카메라들(센터+회전)로부터 LS 삼각측량해서 3D를 만든다.
// 성공하면 X_out에 넣고 true, 아니면 false.
static bool TriangulateTrackFromInitializedCameras(
    const glomap::Track& track,
    const std::unordered_map<image_t, glomap::Image>& images,
    Eigen::Vector3d* X_out)
{
  // 최소 2뷰가 필요
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  int used = 0;

  for (const auto& ob : track.observations) {
    const image_t img_id = ob.first;
    const size_t feat_idx = ob.second;

    auto it = images.find(img_id);
    if (it == images.end()) continue;
    const auto& img = it->second;
    if (!img.is_registered) continue;

    // 관측 벡터 (카메라 좌표계에서 z+ 로 나가는 방향) -> 월드로
    const Eigen::Vector3d& f = img.features_undist[feat_idx];
    if (f.array().isNaN().any()) continue;

    // world ray = R_wc * f
    Eigen::Vector3d d = img.cam_from_world.rotation.inverse() * f;
    const double n = d.norm();
    if (n <= 1e-12) continue;
    d /= n;

    // 이 단계에서는 cam_from_world.translation 이 "카메라 센터 C" 로 세팅돼 있음
    const Eigen::Vector3d C = img.cam_from_world.translation;

    // 선형 LS: (I - d dᵀ)(X - C) = 0  ->  (I - d dᵀ) X = (I - d dᵀ) C
    const Eigen::Matrix3d Ai = Eigen::Matrix3d::Identity() - d * d.transpose();
    A += Ai;
    b += Ai * C;
    ++used;
  }

  if (used < 2) {
    return false;  // 시점이 1개뿐이면 불가
  }

  Eigen::FullPivLU<Eigen::Matrix3d> lu(A);
  if (!lu.isInvertible()) {
    return false;
  }

  Eigen::Vector3d X = lu.solve(b);
  if (!X.allFinite()) {
    return false;
  }

  *X_out = X;
  return true;
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

  const auto saved_constraint = options_.constraint_type;

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
  // InitializeRandomPositions(view_graph, images, tracks);
  // Initialize camera centers according to option (RANDOM or SCALED_FROM_S),
  // and save them in init_centers_.
  InitializeCameraCenters(view_graph, cameras, images, tracks);

  if (options_.dump_init_centers_csv) {
    DumpInitialCentersCSV(images, options_.init_centers_csv_path);
  }

  if (options_.constraint_type == GlobalPositionerOptions::ONLY_SCALEDPOINTS) {
    options_.constraint_type = GlobalPositionerOptions::ONLY_POINTS;
  }

  // Add the camera to camera constraints to the problem.
  if (options_.constraint_type != GlobalPositionerOptions::ONLY_POINTS) {
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

  ceres::Solve(options_.solver_options, problem_.get(), &summary);

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
  }

  ConvertResults(images);
  options_.constraint_type = saved_constraint; // restore
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

void GlobalPositioner::InitializeCameraCenters(
    const ViewGraph& view_graph,
    const std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  init_centers_.clear();

  switch (options_.center_init_mode) {
    case GlobalPositionerOptions::CenterInitMode::RANDOM: {
      // Keep the existing random initialization
      InitializeRandomPositions(view_graph, images, tracks);
      for (auto& [img_id, img] : images)
        init_centers_[img_id] = img.cam_from_world.translation;
      break;
    }

    case GlobalPositionerOptions::CenterInitMode::SCALED_FROM_S: {
    // --- SCALED_FROM_S: use s_i & dir_i to set centers ---
    // Choose root (by inlier sum) & build per-camera world directions
    image_t root_id = SelectRootByInlierSum(view_graph, images);
    c_root_fixed_ = images.at(root_id).Center();
    BuildScaledCamDirectionsTree(view_graph, images, root_id);

    // Estimate s_i (MST + frustum voting refinement)
    InitScalesByMST_FromViewGraphMatches(view_graph, images, tracks, root_id);

    // Bake centers: C_i = C_root + s_i * dir_i
    for (const auto& [img_id, img] : images) {
      auto itS = s_index_.find(img_id);
      auto itD = dir_param_holder_.find(img_id);
      if (itS == s_index_.end() || itD == dir_param_holder_.end()) continue;

      const double s = s_vars_[itS->second];
      const Eigen::Vector3d& dir = itD->second;
      const Eigen::Vector3d Ci = c_root_fixed_ + s * dir;

      images.at(img_id).cam_from_world.translation = Ci; // set init center
      init_centers_[img_id] = Ci;                        // save init center
    }
      break;
    }

    case GlobalPositionerOptions::CenterInitMode::S_ALPHA_LSQ: {
      InitializeCameraCenters_SAlpha(view_graph, cameras, images, tracks);
      break;
    }

    case GlobalPositionerOptions::CenterInitMode::TRI_RANSAC: {
      // 1) Choose root and lock its center.
      image_t root_id = SelectRootByInlierSum(view_graph, images);
      c_root_fixed_ = images.at(root_id).Center();

      // 2) Build per-camera world directions (with random flips) as in your existing method.
      BuildScaledCamDirectionsTree(view_graph, images, root_id);

      // 3) Collect edge scales using triplet RANSAC.
      std::unordered_map<uint64_t, std::vector<double>> edge_scales;
      EstimateEdgeScalesByTriRansac(view_graph, images, tracks, cameras, edge_scales);

      // 4) Convert edge scales to per-camera s_i via MST propagation.
      InitSiFromEdgeScales(images, view_graph, edge_scales, root_id);

      // 5) Bake centers: C_i = C_root + s_i * dir_i
      for (const auto& [img_id, img] : images) {
        auto itS = s_index_.find(img_id);
        auto itD = dir_param_holder_.find(img_id);
        if (itS == s_index_.end() || itD == dir_param_holder_.end()) continue;

        const double s = s_vars_[itS->second];
        const Eigen::Vector3d& dir = itD->second;
        const Eigen::Vector3d Ci = c_root_fixed_ + s * dir;

        images.at(img_id).cam_from_world.translation = Ci;
        init_centers_[img_id] = Ci;
      }
      break;
    }

    case GlobalPositionerOptions::CenterInitMode::LOAD_FROM_CSV: {
      // CSV (image_id,cx,cy,cz) 를 읽어 init_centers_ 채우고 동일하게 세팅
      std::unordered_map<image_t, Eigen::Vector3d> csvC;
      LoadCentersCSV(options_.init_centers_csv_path_in, &csvC);
      for (auto& [img_id, img] : images) {
        auto it = csvC.find(img_id);
        if (it != csvC.end()) {
          img.cam_from_world.translation = it->second;
          init_centers_[img_id]         = it->second;
        } else {
          // fallback: GT 없으면 현재 Center()
          const Eigen::Vector3d C = img.Center();
          img.cam_from_world.translation = C;
          init_centers_[img_id] = C;
        }
      }
      image_t root_id = SelectRootByInlierSum(view_graph, images);
      auto it_root = init_centers_.find(root_id);
      if (it_root != init_centers_.end()) {
        c_root_fixed_ = it_root->second;
      } else {
        // 루트가 CSV에 없으면 첫 항목으로 폴백
        c_root_fixed_ = init_centers_.begin()->second;
      }
      break;
    }
  }
}

void GlobalPositioner::InitializeCameraCenters_SAlpha(
    const ViewGraph& view_graph,
    const std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    const std::unordered_map<track_t, Track>& tracks)
{
  init_centers_.clear();

  // 1) Choose a root by the largest sum of inliers and seed its center.
  image_t root_id = SelectRootByInlierSum(view_graph, images);
  init_centers_[root_id] = images.at(root_id).Center();

  // 2) Build adjacency (weights = inlier counts) and the maximum spanning tree (MST).
  std::unordered_map<image_t, std::vector<std::pair<image_t,int>>> adj;
  int max_w = 0;
  for (const auto& [pair_id, P] : view_graph.image_pairs) {
    if (!P.is_valid) continue;
    const int w = static_cast<int>(P.inliers.size());
    if (w <= 0) continue;
    adj[P.image_id1].push_back({P.image_id2, w});
    adj[P.image_id2].push_back({P.image_id1, w});
    max_w = std::max(max_w, w);
  }
  auto parent = BuildMST_MaxWeight(adj, images, root_id);

  // 3) Build children lists for BFS traversal.
  std::unordered_map<image_t, std::vector<image_t>> children;
  for (const auto& [v, p] : parent) if (v != p) children[p].push_back(v);

  // --- Tunables / guards ---
  const int    MIN_PTS        = 20; // minimum kept samples after gating
  const double EPS            = std::max(1e-12, options_.s_alpha_eps);
  const double MAX_NORM_XY    = 2.0;    // frustum gate in normalized coords
  const double MIN_Z          = 1e-3;     // near plane for gating
  const double MIN_ANGLE_DEG  = 1.0;      // reject near-parallel viewing rays
  const double RMS_MAX        = 500.0;    // LSQ reliability gate
  const double A_ABS_MAX      = 5.0;     // clamp |alpha| per MST edge
  const bool   ANCHOR_FIRST_RING = true;  // enforce |alpha|=1 for direct children of root

  // 4) BFS over the MST.
  std::queue<image_t> q; q.push(root_id);

  // Optional: level map to anchor the first ring around the root.
  std::unordered_map<image_t, int> level;
  level[root_id] = 0;

  while (!q.empty()) {
    image_t u = q.front(); q.pop();
    for (image_t v : children[u]) {

      // -- Find an image pair entry (either direction) to get a relative pose.
      const ImagePair* P_uv = nullptr;
      for (const auto& [pid, P] : view_graph.image_pairs) {
        if (!P.is_valid) continue;
        if ((P.image_id1 == u && P.image_id2 == v) ||
            (P.image_id1 == v && P.image_id2 == u)) {
          P_uv = &P; break;
        }
      }

      // -- Relative pose i->j: R_ij and unit translation in cam-j (uhat_j).
      Eigen::Matrix3d R_ij = Eigen::Matrix3d::Identity();
      Eigen::Vector3d uhat_j(1,0,0);
      if (P_uv) {
        const bool flip = (P_uv->image_id1 != u);
        const Eigen::Quaterniond Rij_q =
            flip ? P_uv->cam2_from_cam1.rotation.conjugate()
                 : P_uv->cam2_from_cam1.rotation;
        const Eigen::Vector3d tij =
            flip ? (-P_uv->cam2_from_cam1.translation)
                 : P_uv->cam2_from_cam1.translation;
        R_ij = Rij_q.toRotationMatrix();
        const double n = tij.norm();
        uhat_j = (n > EPS) ? (tij / n) : Eigen::Vector3d(1,0,0);
      }

      // -- Local camera data and current centers.
      const auto& Imi = images.at(u);
      const auto& Imj = images.at(v);
      const Eigen::Vector3d Ci0 =
          (init_centers_.count(u) ? init_centers_.at(u)
                                  : Imi.cam_from_world.translation);

      const Eigen::Matrix3d Rci = Imi.cam_from_world.rotation.toRotationMatrix();
      const Eigen::Matrix3d Rcj = Imj.cam_from_world.rotation.toRotationMatrix();
      const Eigen::Quaterniond& Rcw_i = Imi.cam_from_world.rotation;
      const Eigen::Quaterniond& Rcw_j = Imj.cam_from_world.rotation;

      // -- Preferred world direction for edge (u -> v). If not available, fall back to v optical axis.
      Eigen::Vector3d dir_w;
      if (!GetEdgeWorldDirection(view_graph, images, u, v, &dir_w)) {
        dir_w = Rcj.transpose() * Eigen::Vector3d(0,0,1); // v's optical axis in world
      }
      const double dn = dir_w.norm();
      dir_w = (dn > EPS) ? (dir_w / dn) : Eigen::Vector3d(1,0,0);

      const Eigen::Vector3d Cj0 = Ci0 + dir_w;

      // -- Collect shared world rays (u and v) and perform geometric gating.
      auto rays_raw = CollectSharedRaysWithPixels(u, v, images, tracks, cameras); // (ru, rv, xj_px)
      // 게이팅/삼각화 단계에서 xj_px도 같이 유지
      std::vector<Eigen::Vector3d> Xw_kept;
      std::vector<Eigen::Vector2d> xj_px_kept;
      Xw_kept.reserve(rays_raw.size());
      xj_px_kept.reserve(rays_raw.size());

      int kept = 0;
      for (const auto& rr : rays_raw) {
        Eigen::Vector3d ru = rr.ru_world;
        Eigen::Vector3d rv = rr.rv_world;
        const double nu = ru.norm(), nv = rv.norm();
        if (nu <= EPS || nv <= EPS) continue;
        ru /= nu; rv /= nv;

        double cosang = ru.dot(rv);
        cosang = std::max(-1.0, std::min(1.0, cosang));
        double ang_deg = std::acos(cosang) * 180.0 / M_PI;
        if (ang_deg < MIN_ANGLE_DEG) continue;

        Eigen::Vector3d Xw = TriangulateMidpoint(Ci0, ru, Cj0, rv);
        if (!InFrustumNormalized(Xw, Ci0, Rcw_i, MAX_NORM_XY, MIN_Z)) continue;
        if (!InFrustumNormalized(Xw, Cj0, Rcw_j, MAX_NORM_XY, MIN_Z)) continue;

        Xw_kept.push_back(Xw);
        xj_px_kept.push_back(rr.xj_px);   // ← v의 실제 픽셀 저장
        ++kept;
      }

      // -- If not enough points survive gating, fall back to unit step along dir_w.
      if (kept < MIN_PTS) {
        double a = 1.0;
        if (ANCHOR_FIRST_RING && level[u] == 0) a = 1.0; // keep unit magnitude at first ring
        const Eigen::Vector3d Cv = Ci0 + a * dir_w;
        init_centers_[v] = Cv;
        level[v] = level[u] + 1;
        q.push(v);
        LOG(INFO) << "[S_ALPHA] edge " << u << "->" << v
                  << " kept=" << kept << "/" << rays_raw.size()
                  << " -> fallback a=1";
        continue;
      }

      // -- Build Xi, Xj in respective camera frames from kept 3D points.
      // intrinsics for j
      double fx_j, fy_j, cx_j, cy_j;
      {
        const camera_t cid_j = Imj.camera_id;
        const Camera& cam_j = cameras.at(cid_j);
        GetIntrinsics(cam_j, &fx_j, &fy_j, &cx_j, &cy_j); // 이미 유틸 있음
      }

      // Xi, Xj_cam 만들기
      std::vector<Eigen::Vector3d> Xi, Xj_cam;
      Xi.reserve(Xw_kept.size());
      Xj_cam.reserve(Xw_kept.size());
      for (const auto& Xw : Xw_kept) {
        Xi.emplace_back(Rci * (Xw - Ci0));
        Xj_cam.emplace_back(Rcj * (Xw - Cj0));
      }

      // (이제 xj_px는 xj_px_kept 바로 사용)
      double s, alpha, rms;
      bool ok_sa = EdgeRansacSolveSAlpha(
                    Xi, Xj_cam, R_ij, uhat_j, options_,
                    &xj_px_kept, fx_j, fy_j, cx_j, cy_j,
                    &s, &alpha, &rms);

      // -- Choose edge length 'a' with reliability/clamp and first-ring anchor.
      double a = 1.0;
      const bool reliable = ok_sa && std::isfinite(rms) && (rms < RMS_MAX);
      if (reliable && std::isfinite(alpha) && std::abs(alpha) > 1e-12) {
        a = std::clamp(alpha, -A_ABS_MAX, A_ABS_MAX);
      } else {
        // unreliable → use sign only; if NaN or zero, use +1
        a = (std::isfinite(alpha) && alpha < 0.0) ? -1.0 : 1.0;
      }
      if (ANCHOR_FIRST_RING && level[u] == 0) {
        // keep only the sign for direct children of the root
        a = (a >= 0.0) ? 1.0 : -1.0;
      }

      // -- Accumulate center: C_v = C_u + a * dir_w.
      const Eigen::Vector3d Cv = Ci0 + a * dir_w;
      init_centers_[v] = Cv;
      level[v] = level[u] + 1;
      q.push(v);

      LOG(INFO) << "[S_ALPHA] edge " << u << "->" << v
                << " Nraw=" << rays_raw.size()
                << " kept=" << kept
                << " rms=" << rms
                << " alpha=" << alpha
                << " a_used=" << a;
    }
  }

  // 5) Bake results into images.
  for (auto& [img_id, img] : images) {
    auto it = init_centers_.find(img_id);
    if (it != init_centers_.end()) {
      img.cam_from_world.translation = it->second;
    }
  }

  // Debug summary
  LOG(INFO) << "[S_ALPHA] adj nodes=" << adj.size();
  size_t child_edges = 0;
  for (auto& kv : children) child_edges += kv.second.size();
  LOG(INFO) << "[S_ALPHA] total children edges=" << child_edges;
}

// ========================= TRIPLET RANSAC (single triplet) =========================
//
// Estimate [s_ij, s_jk, s_ik] for a triplet (i,j,k) using RANSAC+angular reprojection error.
// Returns true and fills s_out (unit-norm with sign convention s_jk >= 0).
//
bool GlobalPositioner::EstimateSForTripletRansac(
  image_t i, image_t j, image_t k,
  const std::vector<track_t>& tids,                // tracks observed by all three views
  const ViewGraph& view_graph,
  const std::unordered_map<image_t, Image>& images,
  const std::unordered_map<track_t, Track>& tracks,
  const std::unordered_map<camera_t, Camera>& cameras,
  Eigen::Vector3d* s_out)
{
  // Pre-fetch relative rotations and unit translations.
  Eigen::Matrix3d Rij, Rjk, Rik;
  Eigen::Vector3d t_hat_ij, t_hat_jk, t_hat_ik;

  if (!GetRelPose_I_to_J(view_graph, i, j, &Rij, &t_hat_ij)) return false;
  if (!GetRelPose_I_to_J(view_graph, j, k, &Rjk, &t_hat_jk)) return false;
  if (!GetRelPose_I_to_J(view_graph, i, k, &Rik, &t_hat_ik)) return false;

  auto build_A = [&](track_t tid)->Eigen::Matrix3d {
    // Build 3x3 A for a single point (point observed by i,j,k).
    // A * s = 0 with s = [s_ij, s_jk, s_ik]^T
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

  // --- Intrinsics (fx=fy) and E/F per pair ---
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


  // 1) ANGULAR: sum of three angular residuals (radians), per-view threshold (deg)
  const double ang_thr = Deg2Rad(options_.tri_inlier_ang_thresh_deg);
  auto angular_sum = [&](track_t tid, const Eigen::Vector3d& s, double* out_sum)->bool {
    const int fi = FindObs(tracks.at(tid), i);
    const int fj = FindObs(tracks.at(tid), j);
    const int fk = FindObs(tracks.at(tid), k);
    if (fi < 0 || fj < 0 || fk < 0) return false;

    const Eigen::Vector3d di = NormalizeSafe(images.at(i).features_undist[fi]);
    const Eigen::Vector3d dj = NormalizeSafe(images.at(j).features_undist[fj]);
    const Eigen::Vector3d dk = NormalizeSafe(images.at(k).features_undist[fk]);

    // Build sized translations
    const Eigen::Vector3d tij = s(0) * t_hat_ij; // in j
    const Eigen::Vector3d tjk = s(1) * t_hat_jk; // in k

    // Depth solve
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

  // 2) PIXEL_REPROJ: sum of three per-pair pixel L2 errors (squared)
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

    // Observed pixels
    const Eigen::Vector3d xi = RayToPixH(di, fix,fiy,cix,ciy);
    const Eigen::Vector3d xj = RayToPixH(dj, fjx,fjy,cjx,cjy);
    const Eigen::Vector3d xk = RayToPixH(dk, fkx,fky,ckx,cky);

    // Sized translations
    const Eigen::Vector3d tij = s(0) * t_hat_ij; // in j
    const Eigen::Vector3d tjk = s(1) * t_hat_jk; // in k

    // Solve depths (same as angular)
    Eigen::Matrix<double,6,3> A;
    Eigen::Matrix<double,6,1> b;
    A.block<3,1>(0,0) = -(Rij * di); A.block<3,1>(0,1) = dj; A.block<3,1>(0,2) = Eigen::Vector3d::Zero();
    b.segment<3>(0)    =  tij;
    A.block<3,1>(3,0) =  Eigen::Vector3d::Zero(); A.block<3,1>(3,1) = -(Rjk * dj); A.block<3,1>(3,2) = dk;
    b.segment<3>(3)    =  tjk;

    const Eigen::Vector3d lambda = A.colPivHouseholderQr().solve(b);
    if (!lambda.allFinite()) return false;
    if (lambda(0)<=options_.tri_min_depth || lambda(1)<=options_.tri_min_depth || lambda(2)<=options_.tri_min_depth) return false;

    // Project predicted points to pixels and compute per-pair pixel error
    const Eigen::Vector3d Xi = lambda(0)*di;                    // cam-i
    const Eigen::Vector3d Xj = Rij*Xi + tij;                    // cam-j
    const Eigen::Vector3d Xk = Rjk*Xj + tjk;                    // cam-k

    const Eigen::Vector3d xj_pred = RayToPixH(Xj, fjx,fjy,cjx,cjy);
    const Eigen::Vector3d xk_pred = RayToPixH(Xk, fkx,fky,ckx,cky);
    const Eigen::Vector3d xi_pred = RayToPixH(Xi, fix,fiy,cix,ciy);

    const double e_ij = (xj.head<2>() - xj_pred.head<2>()).squaredNorm();
    const double e_jk = (xk.head<2>() - xk_pred.head<2>()).squaredNorm();
    const double e_ik = (xi.head<2>() - xi_pred.head<2>()).squaredNorm();
    *out_sum = e_ij + e_jk + e_ik;
    return std::isfinite(*out_sum);
  };

  // 3) SAMPSON: sum of three Sampson squared errors (pixels^2 approx)
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

  // Select consensus function and threshold based on options
  std::function<bool(track_t, const Eigen::Vector3d&, double*)> consensus_fn;
  double inlier_thr_sum = 0.0;

  switch (options_.tri_consensus_metric) {
    case GlobalPositionerOptions::TriConsensusMetric::ANGULAR:
      consensus_fn = angular_sum;
      inlier_thr_sum = 3.0 * ang_thr;               // sum of per-view angles
      break;
    case GlobalPositionerOptions::TriConsensusMetric::PIXEL_REPROJ:
      consensus_fn = pixel_sum;
      inlier_thr_sum = px_thr_sum_sq;               // sum of 3 pairwise squared px
      break;
    case GlobalPositionerOptions::TriConsensusMetric::SAMPSON:
      consensus_fn = sampson_sum;
      inlier_thr_sum = sam_thr_sum_sq;              // sum of 3 Sampson errors
      break;
  }

  // RANSAC loop 
  std::mt19937 rng(options_.seed);
  if (tids.empty()) return false;
  std::uniform_int_distribution<int> uni(0, static_cast<int>(tids.size()) - 1);

  int best_inl = -1;
  Eigen::Vector3d best_s(1,0,0);

  for (int it = 0; it < options_.tri_ransac_max_iters; ++it) {
    const track_t t0 = tids[uni(rng)];  // minimal sample: 1 point

    // Model from minimal sample via SVD(A)
    Eigen::Matrix3d Am = build_A(t0);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Am, Eigen::ComputeFullV);
    Eigen::Vector3d s = svd.matrixV().col(2); // smallest singular vector
    if (!s.allFinite() || s.norm() < 1e-12) continue;
    s.normalize();

    // Fix gauge sign so that s_jk >= 0 for consistency
    if (s(1) < 0) s = -s;

    // Consensus
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

  // LSQ re-fit with inliers
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

// ========================= COLLECT EDGE SCALES VIA TRIPLET RANSAC =========================
//
// For many triplets (i,j,k), estimate [s_ij, s_jk, s_ik] and collect them into edge buckets.
// edge_scales key = (min(i,j)<<32) | max(i,j), values = multiple estimates.
//
void GlobalPositioner::EstimateEdgeScalesByTriRansac(
  const ViewGraph& view_graph,
  const std::unordered_map<image_t, Image>& images,
  const std::unordered_map<track_t, Track>& tracks,
  const std::unordered_map<camera_t, Camera>& cameras,
  std::unordered_map<uint64_t, std::vector<double>>& edge_scales)
{
  // 1) Build per-track observation lists with >=3 views
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

  // 2) Build (i,j,k) -> list of tracks seen by all three
  struct TriKey { image_t i, j, k;
    bool operator==(const TriKey& o) const { return i==o.i && j==o.j && k==o.k; } };
  struct TriKeyHash { size_t operator()(const TriKey& t) const {
    return (size_t)t.i*1315423911u ^ (size_t)t.j*2654435761u ^ (size_t)t.k; } };
  
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

  std::vector<int> sizes;
  sizes.reserve(tri_points.size());
  for (const auto& kv : tri_points) sizes.push_back((int)kv.second.size());

  if (!sizes.empty()) {
    // 정렬 후 통계치
    std::sort(sizes.begin(), sizes.end());
    const int N = (int)sizes.size();
    double sum = 0;
    for (int s : sizes) sum += s;
    const double mean = sum / N;
    const int median = sizes[N/2];
    const int p10 = sizes[(int)(0.10 * (N-1))];
    const int p90 = sizes[(int)(0.90 * (N-1))];

    LOG(INFO) << "[TRI_STATS] triplets=" << N
              << " mean=" << mean
              << " median=" << median
              << " p10=" << p10
              << " p90=" << p90;

    // 간단한 히스토그램(bin=1)
    std::unordered_map<int,int> hist;
    for (int s : sizes) ++hist[s];
    // 몇 개만 출력
    int shown = 0;
    LOG(INFO) << "[TRI_STATS] histogram (tracks-per-triplet -> count):";
    for (const auto& kv : hist) {
      if (shown++ > 20) { LOG(INFO) << " ..."; break; }
      LOG(INFO) << "  " << kv.first << " -> " << kv.second;
    }

    // 트랙이 특히 많은 top-5 트리플릿 샘플
    LOG(INFO) << "[TRI_STATS] top-5 triplets by shared tracks:";
    std::vector<std::pair<TriKey,int>> ranked;
    ranked.reserve(tri_points.size());
    for (const auto& kv : tri_points) ranked.push_back({kv.first, (int)kv.second.size()});
    std::partial_sort(ranked.begin(), ranked.begin()+std::min(5,(int)ranked.size()), ranked.end(),
                      [](auto& a, auto& b){ return a.second > b.second; });
    for (int i=0;i<std::min(5,(int)ranked.size());++i) {
      const auto& key = ranked[i].first;
      LOG(INFO) << "  (" << key.i << "," << key.j << "," << key.k << ") : " << ranked[i].second;
    }
  } else {
    LOG(INFO) << "[TRI_STATS] tri_points is empty.";
  }

  // 3) For each triplet, run RANSAC and collect scales for edges.
  auto push_edge = [&](image_t a, image_t b, double s_ab){
    image_t u = std::min(a,b), v = std::max(a,b);
    uint64_t k = (uint64_t(u) << 32) | uint64_t(v);
    edge_scales[k].push_back(std::abs(s_ab)); // store magnitude; sign handled later by direction
  };

  int accepted = 0;
  for (const auto& [key, tids] : tri_points) {
    if (static_cast<int>(tids.size()) < options_.tri_min_inliers) continue;

    Eigen::Vector3d s_hat;
    if (!EstimateSForTripletRansac(key.i, key.j, key.k, tids,
                                  view_graph, images, tracks, cameras, &s_hat)) {
      continue;
    }
    ++accepted;

    // s_hat = [s_ij, s_jk, s_ik]
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

// ========================= EDGE->PER-CAMERA s_i VIA MST =========================
//
// Aggregate multiple edge scales by median, then propagate along MST to get s_i.
// Direction sign is determined by child dir and world edge direction.
//
void GlobalPositioner::InitSiFromEdgeScales(
  const std::unordered_map<image_t, Image>& images,
  const ViewGraph& view_graph,
  const std::unordered_map<uint64_t, std::vector<double>>& edge_scales,
  image_t root_id)
{
  // Prepare s_vars_ / s_index_
  s_vars_.assign(images.size(), 0.0);
  s_index_.clear();
  { size_t idx=0; for (const auto& kv : images) s_index_[kv.first] = idx++; }
  s_vars_[s_index_.at(root_id)] = 0.0;

  // Build adjacency (weights by inliers count from view_graph)
  std::unordered_map<image_t, std::vector<std::pair<image_t,int>>> adj;
  for (const auto& [pid, P] : view_graph.image_pairs) {
    if (!P.is_valid) continue;
    const int w = static_cast<int>(P.inliers.size());
    if (w <= 0) continue;
    adj[P.image_id1].push_back({P.image_id2, w});
    adj[P.image_id2].push_back({P.image_id1, w});
  }
  auto parent = BuildMST_MaxWeight(adj, images, root_id);

  // children list
  std::unordered_map<image_t, std::vector<image_t>> children;
  for (const auto& [v,p] : parent) if (v!=p) children[p].push_back(v);

  auto edge_key = [](image_t a,image_t b)->uint64_t{
    image_t u = std::min(a,b), v = std::max(a,b);
    return (uint64_t(u)<<32) | uint64_t(v);
  };

  std::queue<image_t> q; q.push(root_id);
  while (!q.empty()) {
    image_t u = q.front(); q.pop();
    for (image_t v : children[u]) {
      // median aggregation
      double s_uv = 1.0;
      const uint64_t ek = edge_key(u,v);
      auto it = edge_scales.find(ek);
      if (it != edge_scales.end() && !it->second.empty()) {
        std::vector<double> vec = it->second;
        std::nth_element(vec.begin(), vec.begin() + vec.size()/2, vec.end());
        s_uv = vec[vec.size()/2];
      }

      // world edge direction (u->v); fallback to v optical axis if missing
      Eigen::Vector3d dir_edge_world;
      if (!GetEdgeWorldDirection(view_graph, images, u, v, &dir_edge_world)) {
        const Eigen::Matrix3d RvT = images.at(v).cam_from_world.rotation.toRotationMatrix().transpose();
        dir_edge_world = NormalizeSafe(RvT * Eigen::Vector3d(0,0,1));
      }

      // sign from child direction
      const Eigen::Vector3d& dir_v = dir_param_holder_.at(v);
      const double sign = (dir_v.dot(dir_edge_world) >= 0.0) ? 1.0 : -1.0;

      const double a_uv = sign * s_uv; // projected step
      const double s_u  = s_vars_[s_index_.at(u)];
      s_vars_[s_index_.at(v)] = s_u + a_uv;

      q.push(v);
    }
  }
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

  if (!options_.generate_random_positions || !options_.optimize_positions) {
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
  LOG(INFO) << "[Init RANDOM] generate_random_positions=" << options_.generate_random_positions
        << " optimize_positions=" << options_.optimize_positions
        << " constrained=" << constrained_positions.size();
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

      if (options_.optimize_points && !track.is_initialized) {
        Eigen::Vector3d Xtri;
        if (TriangulateTrackFromInitializedCameras(track, images, &Xtri)) {
          track.xyz = Xtri;
          track.is_initialized = true;
        }
      }

      // 2) 삼각측량이 안 됐고, 예전처럼 랜덤도 허용되어 있으면 랜덤
      if (options_.optimize_points &&
          !track.is_initialized &&        // 위에서 실패한 경우
          options_.generate_random_points) {
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

    // 2) set global root center (choose one policy)
    // simplest: center
    c_root_fixed_ = images.at(root_id).Center();

    // build per-camera world directions (rooted)
    BuildScaledCamDirectionsTree(view_graph, images, root_id);

    // initialize s_i using MST weighted by match counts (preferred)
    // (If your ImagePair uses a different field than `num_inlier_matches`, rename there.)
    InitScalesByMST_FromViewGraphMatches(view_graph, images, tracks, root_id);

    // 3) preallocate s_i with stable addresses, init to 1.0
    // EnforceOutwardDirs(images);

    // Uniform 1 scale setting
    // s_vars_.clear();
    // s_index_.clear();
    // s_vars_.assign(images.size(), 1.0);

    // // Random scale setting
    // s_vars_.resize(images.size());            // fixed capacity & addresses
    // for (auto &v : s_vars_){
    //   v = 100.0 * RandomDouble(random_generator_, -1, 1);
    // }

    size_t idx = 0;
    for (const auto& [image_id, image] : images) {
      s_index_[image_id] = idx++;
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

    // DumpInitScalesCSV(cameras, images, "init_scales.csv");
  }
}

// void GlobalPositioner::EnforceOutwardDirs(const std::unordered_map<image_t, Image>& images) {
//   if (s_index_.empty()) return;

//   const Eigen::Vector3d Croot = c_root_fixed_;
//   int flipped = 0;

//   for (const auto& [img_id, img] : images) {
//     auto it_dir = dir_param_holder_.find(img_id);
//     auto it_si  = s_index_.find(img_id);
//     if (it_dir == dir_param_holder_.end() || it_si == s_index_.end()) continue;

//     Eigen::Vector3d& dir = it_dir->second;
//     double& s            = s_vars_[it_si->second];

//     // 현재 카메라 센터
//     const Eigen::Vector3d Ci = Croot + s * dir;

//     // 바깥(=Croot에서 Ci로 향하는 벡터)과 dir의 내적이 음수면 뒤집기
//     if ((Ci - Croot).dot(dir) < 0.0) {
//       dir = -dir;
//       s   = -s;
//       ++flipped;
//     }
//   }
//   LOG(INFO) << "[Init] Outward dir enforcement flipped " << flipped << " cameras.";
// }


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
  bool random_flip_dirs = false;

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

}

void GlobalPositioner::DumpInitialCentersCSV(
    const std::unordered_map<image_t, Image>& images,
    const std::string& csv_path) const {
  std::ofstream csv(csv_path, std::ios::out);
  csv << "image_id,cx,cy,cz\n";
  for (const auto& [img_id, img] : images) {
    auto it = init_centers_.find(img_id);
    if (it == init_centers_.end()) continue;
    const auto& C = it->second;
    csv << img_id << "," << C.x() << "," << C.y() << "," << C.z() << "\n";
  }
}


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
      const double max_norm_xy = 1.9;
      const double min_z       = 1e-4;
      // Scale up the span
      const int    N_samples   = 101;      // was 101
      const double span        = 5.0;      // was 0.5
      const bool   both_cameras   = true;

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
