#include "glomap/estimators/global_pnp.h"
#include <algorithm>

#include <colmap/estimators/pose.h>
#include <colmap/util/logging.h>

namespace glomap {

GlobalPnP::GlobalPnP(const GlobalPnPOptions& options)
    : options_(options) {}

// -----------------------------------------------------------------------------
// Helper: collect 2D–3D correspondences for a given image
// -----------------------------------------------------------------------------
namespace {

void Collect2D3DCorrespondences(
    const image_t image_id,
    const Image& image,
    const std::unordered_map<track_t, Track>& tracks,
    const Camera& camera,
    std::vector<Eigen::Vector2d>* points2D,
    std::vector<Eigen::Vector3d>* points3D) {
  points2D->clear();
  points3D->clear();

  if (image.features.empty()) {
    return;
  }

  for (const auto& kv : tracks) {
    const Track& tr = kv.second;

    // Use only tracks with valid 3D position
    if (!tr.xyz.allFinite()) {
      continue;
    }

    for (const auto& obs : tr.observations) {
      if (obs.first != image_id) {
        continue;
      }

      const size_t feat_id = static_cast<size_t>(obs.second);
      if (feat_id >= image.features.size()) {
        continue;
      }

      // 2D point in image (pixel) coordinates, consistent with colmap::Camera
      const Eigen::Vector2d pt2D = image.features[feat_id];

      points2D->push_back(pt2D);
      points3D->push_back(tr.xyz);
    }
  }
}

}  // namespace

// -----------------------------------------------------------------------------
// Estimate pose of a single image using PnP (COLMAP AbsolutePose)
// -----------------------------------------------------------------------------
bool GlobalPnP::EstimateImagePosePnP(
    const image_t image_id,
    const Camera& camera,
    const Image& image,
    const std::unordered_map<track_t, Track>& tracks,
    colmap::Rigid3d* cam_from_world_out) const {
  std::vector<Eigen::Vector2d> points2D;
  std::vector<Eigen::Vector3d> points3D;

  Collect2D3DCorrespondences(image_id, image, tracks, camera,
                             &points2D, &points3D);

  LOG(INFO) << "[GlobalPnP] image " << image_id
            << " PnP start: " << points2D.size()
            << " 2D-3D matches.";

  if (points2D.size() < static_cast<size_t>(options_.min_num_correspondences)) {
    LOG(INFO) << "[GlobalPnP] image " << image_id
              << " has too few 2D-3D matches: " << points2D.size();
    return false;
  }

  colmap::AbsolutePoseEstimationOptions est_opts;
  const double ransac_max_error_px =
      (options_.ransac_max_error_px > 0.0) ? options_.ransac_max_error_px
                                           : 12.0;

  est_opts.ransac_options.max_error        = ransac_max_error_px;
  est_opts.ransac_options.min_inlier_ratio = options_.min_inlier_ratio;
  est_opts.ransac_options.confidence       = options_.ransac_confidence;
  est_opts.ransac_options.min_num_trials   = options_.min_num_ransac_trials;
  est_opts.ransac_options.max_num_trials   = options_.max_num_iterations;

  // 이건 colmap::Camera 메서드
  const double cam_thresh =
      camera.CamFromImgThreshold(est_opts.ransac_options.max_error);

  std::vector<char> inlier_mask;
  size_t num_inliers = 0;

  // Initial pose (identity)
  colmap::Rigid3d cam_from_world_init;

  // Copy glomap Camera into a COLMAP Camera
  colmap::Camera col_camera = camera;

  const bool success = colmap::EstimateAbsolutePose(
      est_opts,
      points2D,
      points3D,
      &cam_from_world_init,   // 4: Rigid3d*
      &col_camera,            // 5: Camera*
      &num_inliers,           // 6: #inliers
      &inlier_mask);          // 7: inlier mask

  if (!success) {
    VLOG(2) << "[GlobalPnP] EstimateAbsolutePose failed for image "
            << image_id;
    return false;
  }

  const double inlier_ratio =
        static_cast<double>(num_inliers) /
        static_cast<double>(points2D.size());

  VLOG(2) << "[GlobalPnP] image " << image_id
          << " PnP success: inliers = " << num_inliers
          << " / " << points2D.size()
          << " (ratio=" << inlier_ratio << ")";

  if (num_inliers < static_cast<size_t>(options_.min_num_correspondences) ||
    inlier_ratio < options_.min_inlier_ratio) {
    VLOG(2) << "[GlobalPnP] image " << image_id
            << " rejected: inliers = " << num_inliers
            << " / " << points2D.size()
            << " (ratio = " << inlier_ratio << ")";
    return false;
  }

  if (options_.refine_pose) {
    colmap::AbsolutePoseRefinementOptions ref_opts;
    ref_opts.max_num_iterations    = 50;      // similar scale to BA
    ref_opts.refine_focal_length   = options_.refine_focal_length;
    // Note: Some COLMAP versions do not have refine_principal_point.
    ref_opts.refine_extra_params   = options_.refine_extra_params;

    colmap::RefineAbsolutePose(
        ref_opts,
        inlier_mask,
        points2D,
        points3D,
        &cam_from_world_init,  // Rigid3d*
        &col_camera,           // Camera*
        nullptr);              // covariance (unused)
  }

  *cam_from_world_out = cam_from_world_init;

  // --- reprojection error (debug) ---
  double avg_err = 0.0;
  int cnt = 0;

  // 평균 focal length (px 단위로 환산하기 위함)
  double avg_focal = 1.0;
  {
    double f_sum = 0.0;
    int f_cnt = 0;
    for (auto idx : col_camera.FocalLengthIdxs()) {
      f_sum += col_camera.params[idx];
      ++f_cnt;
    }
    if (f_cnt > 0) {
      avg_focal = f_sum / f_cnt;
    }
  }

  for (size_t i = 0; i < points2D.size(); ++i) {
    if (!inlier_mask[i]) continue;

    const Eigen::Vector3d& X = points3D[i];

    // 3D -> 카메라 좌표계
    Eigen::Vector3d x_cam = cam_from_world_init.rotation * X +
                            cam_from_world_init.translation;

    // 예측된 정규화 2D (카메라 좌표 / z)
    Eigen::Vector2d proj_norm(x_cam.x() / x_cam.z(),
                              x_cam.y() / x_cam.z());

    // 관측 픽셀을 정규화 좌표로
    Eigen::Vector2d obs_norm = col_camera.CamFromImg(points2D[i]);

    // 정규화 공간에서의 오차
    double err_norm = (proj_norm - obs_norm).norm();

    // focal을 곱해서 대략적인 pixel reprojection error로 환산
    avg_err += err_norm * avg_focal;
    ++cnt;
  }

  if (cnt > 0) {
    avg_err /= cnt;
    LOG(INFO) << "[GlobalPnP] image " << image_id
              << " avg reproj err (inliers only) = "
              << avg_err << " px (approx)";
  }

    return true;
  }

// -----------------------------------------------------------------------------
// Register all orphan images using PnP and mark them as registered
// -----------------------------------------------------------------------------
bool GlobalPnP::Solve(
    const ViewGraph& view_graph,
    const std::vector<image_t>& orphan_images,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    const std::unordered_map<track_t, Track>& tracks) {
  (void)view_graph;

  if (orphan_images.empty()) {
    VLOG(2) << "[GlobalPnP] No orphan images to register.";
    return true;
  }

  int num_success = 0;
  int num_attempt = 0;

  for (const image_t img_id : orphan_images) {
    auto it_img = images.find(img_id);
    if (it_img == images.end()) {
      LOG(INFO) << "[GlobalPnP] orphan image " << img_id
                << " not found in images map.";
      continue;
    }

    Image& image = it_img->second;

    auto it_cam = cameras.find(image.camera_id);
    if (it_cam == cameras.end()) {
      LOG(INFO) << "[GlobalPnP] orphan image " << img_id
                << " has no camera " << image.camera_id;
      continue;
    }

    const Camera& camera = it_cam->second;

    // DEBUG: count raw 2D-3D correspondences
    std::vector<Eigen::Vector2d> dbg_pts2D;
    std::vector<Eigen::Vector3d> dbg_pts3D;
    Collect2D3DCorrespondences(img_id, image, tracks, camera,
                               &dbg_pts2D, &dbg_pts3D);

    LOG(INFO) << "[GlobalPnP] image " << img_id
              << " has " << dbg_pts2D.size()
              << " raw 2D-3D correspondences.";

    if (dbg_pts2D.size() < static_cast<size_t>(options_.min_num_correspondences)) {
      VLOG(2) << "[GlobalPnP] image " << img_id
                << " skipped before PnP: too few matches ("
                << dbg_pts2D.size() << ").";
      continue;
    }

    colmap::Rigid3d cam_from_world_new;
    ++num_attempt;

    if (!EstimateImagePosePnP(img_id, camera, image, tracks,
                              &cam_from_world_new)) {
      LOG(INFO) << "[GlobalPnP] PnP failed for image " << img_id;
      continue;
    }

    image.cam_from_world = cam_from_world_new;
    image.is_registered  = true;
    ++num_success;
  }

  LOG(INFO) << "[GlobalPnP] Registered " << num_success
            << " / " << num_attempt
            << " orphan images via PnP.";

  return true;
}

}  // namespace glomap
