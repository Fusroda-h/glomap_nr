#pragma once

#include "glomap/scene/types_sfm.h"
#include "glomap/types.h"

#include <vector>
#include <unordered_map>

namespace colmap {
class Rigid3d;  // forward declaration
}

namespace glomap {

struct GlobalPnPOptions {
  // Minimum number of 2D–3D correspondences required for PnP.
  int min_num_correspondences = 50;  // abs_pose_min_num_inliers

  // RANSAC inlier threshold in pixels.
  double ransac_max_error_px = 12.0;  // abs_pose_max_error

  // Minimum inlier ratio to accept a pose.
  double min_inlier_ratio = 0.05;  // abs_pose_min_inlier_ratio

  // Maximum number of RANSAC iterations (trials).
  int max_num_iterations = 10000;  // abs_pose_max_num_trials (roughly)

  // RANSAC confidence.
  double ransac_confidence = 0.999;

  // Minimum number of RANSAC iterations (usually small).
  int min_num_ransac_trials = 30;  // abs_pose_num_min_ransac_iterations

  // Whether to run nonlinear refinement after RANSAC.
  bool refine_pose = true;

  // Whether to refine focal length during pose refinement.
  bool refine_focal_length = false;  // COLMAP incremental pipeline도 보통 false

  // Whether to refine extra camera parameters during pose refinement.
  bool refine_extra_params = false;
};

class GlobalPnP {
 public:
  explicit GlobalPnP(const GlobalPnPOptions& options);

  bool Solve(const ViewGraph& view_graph,
             const std::vector<image_t>& orphan_images,
             std::unordered_map<camera_t, Camera>& cameras,
             std::unordered_map<image_t, Image>& images,
             const std::unordered_map<track_t, Track>& tracks);

 private:
  bool EstimateImagePosePnP(
      image_t image_id,
      const Camera& camera,
      const Image& image,
      const std::unordered_map<track_t, Track>& tracks,
      colmap::Rigid3d* cam_from_world_out) const;

  GlobalPnPOptions options_;
};

}  // namespace glomap
