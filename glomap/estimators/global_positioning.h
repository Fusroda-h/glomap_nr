#pragma once

#include <unordered_map>
#include <vector>
#include <cstdint>
#include <memory>
#include <random>

#include "glomap/estimators/optimization_base.h"
#include "glomap/scene/types_sfm.h"
#include "glomap/types.h"

namespace glomap {

struct GlobalPositionerOptions : public OptimizationBaseOptions {
  // Which constraints to use in global positioning.
  enum ConstraintType {
    // Use only point-to-camera (camera-to-point) constraints.
    ONLY_POINTS,
    // Use only camera-to-camera constraints.
    ONLY_CAMERAS,
    // Reweight points and cameras to have similar total contribution.
    POINTS_AND_CAMERAS_BALANCED,
    // Use both point and camera constraints equally.
    POINTS_AND_CAMERAS,
  };

  // Which consensus metric to use when running triplet RANSAC.
  // Must match the switch statement in global_positioning.cc.
  enum TriConsensusMetric {
    ANGULAR = 0,
    PIXEL_REPROJ = 1,
    SAMPSON = 2
  };

  // ---------------------------------------------------------------------------
  // General options
  // ---------------------------------------------------------------------------

  // Whether to generate random camera centers before optimization.
  bool generate_random_positions = true;

  // Whether to generate random 3D points before optimization.
  bool generate_random_points = true;

  // Whether to generate scale variables (default: 1.0).
  bool generate_scales = true;

  // Which variables to optimize during global positioning.
  bool optimize_positions = true;
  bool optimize_points = true;
  bool optimize_scales = true;

  // Minimum number of views required for a track to be used.
  int min_num_view_per_track = 3;

  // Random seed.
  unsigned seed = 1;

  // Which constraint type to use.
  ConstraintType constraint_type = ONLY_POINTS;

  // Reweight factor for POINTS_AND_CAMERAS_BALANCED.
  double constraint_reweight_scale = 1.0;

  // ---------------------------------------------------------------------------
  // Triplet-RANSAC-specific options (used in ONLY_POINTS initialization path)
  // ---------------------------------------------------------------------------

  // Which error to use when scoring a triplet hypothesis.
  TriConsensusMetric tri_consensus_metric = PIXEL_REPROJ;

  // Maximum RANSAC iterations per triplet.
  int tri_ransac_max_iters = 100;

  // Minimum number of inliers for accepting a triplet model.
  int tri_min_inliers = 8;

  // Maximum number of (i,j,k) triplets emitted per track.
  int tri_max_triplets_per_track = 16;

  // Minimum depth during 3-view depth solve.
  double tri_min_depth = 1e-6;

  // Inlier threshold for ANGULAR metric (degrees, per view).
  double tri_inlier_ang_thresh_deg = 3.0;

  // Inlier threshold for PIXEL_REPROJ / SAMPSON (pixels, per pair/view).
  double tri_inlier_px_thresh = 3.0;

  GlobalPositionerOptions() : OptimizationBaseOptions() {
    // Default robust loss for global positioning.
    thres_loss_function = 1e-1;
    loss_function = std::make_shared<ceres::HuberLoss>(thres_loss_function);
  }
};

class GlobalPositioner {
 public:
  explicit GlobalPositioner(const GlobalPositionerOptions& options);

  // Main entry. Returns true on success.
  bool Solve(const ViewGraph& view_graph,
             std::unordered_map<camera_t, Camera>& cameras,
             std::unordered_map<image_t, Image>& images,
             std::unordered_map<track_t, Track>& tracks);

  GlobalPositionerOptions& GetOptions() { return options_; }

 protected:
  // ---------------------------------------------------------------------------
  // Original GLOMAP pipeline pieces
  // ---------------------------------------------------------------------------

  // Create the global Ceres problem and pre-allocate scale variables.
  void SetupProblem(const ViewGraph& view_graph,
                    const std::unordered_map<track_t, Track>& tracks);

  // Initialize camera centers (in world coordinates).
  void InitializeRandomPositions(const ViewGraph& view_graph,
                                 std::unordered_map<image_t, Image>& images,
                                 std::unordered_map<track_t, Track>& tracks);

  // Add camera-to-camera constraints from relative translations.
  void AddCameraToCameraConstraints(const ViewGraph& view_graph,
                                    std::unordered_map<image_t, Image>& images);

  // Add camera-to-point (bearing) constraints.
  void AddPointToCameraConstraints(
      std::unordered_map<camera_t, Camera>& cameras,
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks);

  // Add a single track to the Ceres problem.
  void AddTrackToProblem(track_t track_id,
                         std::unordered_map<camera_t, Camera>& cameras,
                         std::unordered_map<image_t, Image>& images,
                         std::unordered_map<track_t, Track>& tracks);

  // Set Schur ordering for scales / points / cameras.
  void AddCamerasAndPointsToParameterGroups(
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks);

  // Mark some variables constant depending on user options.
  void ParameterizeVariables(std::unordered_map<image_t, Image>& images,
                             std::unordered_map<track_t, Track>& tracks);

  // Convert camera centers back to camera translations (cam_from_world).
  void ConvertResults(std::unordered_map<image_t, Image>& images);

  void GlobalPositioner::DumpInitialCentersCSV(
    const std::unordered_map<image_t, Image>& images,
    const std::string& csv_path) const;

  // ---------------------------------------------------------------------------
  // New: triplet-based initialization path
  // ---------------------------------------------------------------------------

  // Estimate scales [s_ij, s_jk, s_ik] for a single triplet via 1-point RANSAC.
  bool EstimateSForTripletRansac(
      image_t i, image_t j, image_t k,
      const std::vector<track_t>& tids,
      const ViewGraph& view_graph,
      const std::unordered_map<image_t, Image>& images,
      const std::unordered_map<track_t, Track>& tracks,
      const std::unordered_map<camera_t, Camera>& cameras,
      Eigen::Vector3d* s_out);

  // Collect per-edge scale hypotheses from all triplets (i,j,k).
  void EstimateEdgeScalesByTriRansac(
      const ViewGraph& view_graph,
      const std::unordered_map<image_t, Image>& images,
      const std::unordered_map<track_t, Track>& tracks,
      const std::unordered_map<camera_t, Camera>& cameras,
      std::unordered_map<uint64_t, std::vector<double>>& edge_scales);

  // Place cameras in world using view-graph directions + per-edge scales.
  void InitializeCamerasFromTriScales(
      const ViewGraph& view_graph,
      std::unordered_map<image_t, Image>& images,
      const std::unordered_map<uint64_t, std::vector<double>>& edge_scales);

  // Get world-space direction from src -> dst using the relative translation
  // stored in the view graph (converted from dst camera frame to world frame).
  bool GetWorldDirection(
      const ViewGraph& view_graph,
      const std::unordered_map<image_t, Image>& images,
      image_t src, image_t dst,
      Eigen::Vector3d* dir_world) const;

  // Triangulate / initialize points once camera centers are initialized.
  void InitializePointsFromCameras(
      std::unordered_map<camera_t, Camera>& cameras,
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks);

  // ---------------------------------------------------------------------------

  GlobalPositionerOptions options_;

  // Random generator used for random initialization / RANSAC sampling.
  std::mt19937 random_generator_;

  // Ceres problem for the legacy optimization path.
  std::unique_ptr<ceres::Problem> problem_;

  // Loss functions for point-to-camera constraints (calibrated / uncalibrated).
  std::shared_ptr<ceres::LossFunction> loss_function_ptcam_uncalibrated_;
  std::shared_ptr<ceres::LossFunction> loss_function_ptcam_calibrated_;

  // Auxiliary scale variables, one per residual.
  std::vector<double> scales_;
};

}  // namespace glomap
