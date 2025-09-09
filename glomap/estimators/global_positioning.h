#pragma once

#include "glomap/estimators/optimization_base.h"
#include "glomap/scene/types_sfm.h"
#include "glomap/types.h"

namespace glomap {

struct GlobalPositionerOptions : public OptimizationBaseOptions {
  // ONLY_POINTS is recommended
  enum ConstraintType {
    // only include camera to point constraints
    ONLY_POINTS,
    // only include camera to camera constraints
    ONLY_CAMERAS,
    // the points and cameras are reweighted to have similar total contribution
    POINTS_AND_CAMERAS_BALANCED,
    // treat each contribution from camera to point and camera to camera equally
    POINTS_AND_CAMERAS,
    // only include camera to point constraints but calculating with scaled relpose translation
    ONLY_SCALEDPOINTS,
  };

  // Whether initialize the reconstruction randomly
  bool generate_random_positions = true;
  bool generate_random_points = true;
  bool generate_scales = true;  // Now using fixed 1 as initializaiton

  // Flags for which parameters to optimize
  bool optimize_positions = true;
  bool optimize_points = true;
  bool optimize_scales = true;

  // Constrain the minimum number of views per track
  int min_num_view_per_track = 3;

  // Random seed
  unsigned seed = 1;

  // the type of global positioning
  ConstraintType constraint_type = ONLY_POINTS;
  double constraint_reweight_scale =
      1.0;  // only relevant for POINTS_AND_CAMERAS_BALANCED

  GlobalPositionerOptions() : OptimizationBaseOptions() {
    thres_loss_function = 1e-1;
    loss_function = std::make_shared<ceres::HuberLoss>(thres_loss_function);
  }
};

class GlobalPositioner {
 public:
  GlobalPositioner(const GlobalPositionerOptions& options);

  // Returns true if the optimization was a success, false if there was a
  // failure.
  // Assume tracks here are already filtered
  bool Solve(const ViewGraph& view_graph,
             std::unordered_map<camera_t, Camera>& cameras,
             std::unordered_map<image_t, Image>& images,
             std::unordered_map<track_t, Track>& tracks);

  GlobalPositionerOptions& GetOptions() { return options_; }

 protected:
  void SetupProblem(const ViewGraph& view_graph,
                    const std::unordered_map<track_t, Track>& tracks);

  // Initialize all cameras to be random.
  void InitializeRandomPositions(const ViewGraph& view_graph,
                                 std::unordered_map<image_t, Image>& images,
                                 std::unordered_map<track_t, Track>& tracks);

  // Creates camera to camera constraints from relative translations. (3D)
  void AddCameraToCameraConstraints(const ViewGraph& view_graph,
                                    std::unordered_map<image_t, Image>& images);

  // // Add tracks to the problem
  // void AddPointToCameraConstraints(
  //     std::unordered_map<camera_t, Camera>& cameras,
  //     std::unordered_map<image_t, Image>& images,
  //     std::unordered_map<track_t, Track>& tracks);

  // Add tracks to the problem (point-to-camera constraints).
  // We thread view_graph so ONLY_SCALEDPOINTS can prebuild fixed world directions.
  void AddPointToCameraConstraints(
      std::unordered_map<camera_t, Camera>& cameras,
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      const ViewGraph& view_graph);

  // Add a single track to the problem
  void AddTrackToProblem(track_t track_id,
                         std::unordered_map<camera_t, Camera>& cameras,
                         std::unordered_map<image_t, Image>& images,
                         std::unordered_map<track_t, Track>& tracks);

  // Add a single track to the problem (ONLY_SCALEDPOINTS path).
  // Uses 1-DOF scalar s per camera along a fixed world direction.
  void AddTrackToScaledCamProblem(
      track_t track_id,
      std::unordered_map<camera_t, Camera>& cameras,
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks);
  
  void BuildScaledCamDirectionsTree(
      const ViewGraph& view_graph,
      const std::unordered_map<image_t, Image>& images,
      image_t root_id);

  void InitScalesByMST_FromViewGraphMatches(
      const ViewGraph& view_graph,
      const std::unordered_map<image_t, Image>& images,
      const std::unordered_map<track_t, Track>& tracks,
      image_t root_id);

  void DumpInitScalesCSV(
      const std::unordered_map<camera_t, Camera>& cameras,
      const std::unordered_map<image_t, Image>& images,
      const std::string& csv_path) const;

  void EnforceOutwardDirs(const std::unordered_map<image_t, Image>& images);

  // Set the parameter groups
  void AddCamerasAndPointsToParameterGroups(
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks);

  // Parameterize the variables, set some variables to be constant if desired
  void ParameterizeVariables(std::unordered_map<image_t, Image>& images,
                             std::unordered_map<track_t, Track>& tracks);

  // During the optimization, the camera translation is set to be the camera
  // center Convert the results back to camera poses
  void ConvertResults(std::unordered_map<image_t, Image>& images);

  GlobalPositionerOptions options_;

  std::mt19937 random_generator_;
  std::unique_ptr<ceres::Problem> problem_;

  // Loss functions for reweighted terms.
  std::shared_ptr<ceres::LossFunction> loss_function_ptcam_uncalibrated_;
  std::shared_ptr<ceres::LossFunction> loss_function_ptcam_calibrated_;

  // Auxiliary scale variables.
  std::vector<double> scales_;

  // === ONLY_SCALEDPOINTS state ===
  // global fixed root center for all cameras (default: (0,0,0))
  Eigen::Vector3d c_root_fixed_ = Eigen::Vector3d::Zero();
  // per-camera 1-DOF scalars
  std::vector<double> s_vars_;
  // Map image_id -> index into s_vars_.
  std::unordered_map<image_t, size_t> s_index_;
  // per-camera fixed world directions (stable storage for .data() if needed elsewhere)
  std::unordered_map<image_t, Eigen::Vector3d> dir_param_holder_;
};

}  // namespace glomap
