#include "global_mapper.h"

#include "glomap/processors/image_pair_inliers.h"
#include "glomap/processors/image_undistorter.h"
#include "glomap/processors/reconstruction_pruning.h"
#include "glomap/processors/relpose_filter.h"
#include "glomap/processors/track_filter.h"
#include "glomap/processors/view_graph_manipulation.h"

#include <colmap/util/timer.h>
#include <random>
#include <filesystem>

namespace glomap {

void InjectOutlierEdges(ViewGraph& view_graph,
                        const std::unordered_map<image_t, Image>& images,
                        int num_outliers) {
  if (num_outliers <= 0) return;

  // 1) largest connected component 안의 이미지만 대상으로 사용
  std::vector<image_t> img_ids;
  img_ids.reserve(images.size());
  for (const auto& kv : images) {
    const image_t id = kv.first;
    const Image& img = kv.second;
    if (img.is_registered) {  // 이미 LCC 안에 있는 애들만 사용
      img_ids.push_back(id);
    }
  }
  if (img_ids.size() < 2) return;

  std::mt19937 rng(0);  // 재현성 있게 고정 시드
  std::uniform_int_distribution<size_t> dist(0, img_ids.size() - 1);

  int inserted = 0;
  while (inserted < num_outliers) {
    image_t id1 = img_ids[dist(rng)];
    image_t id2 = img_ids[dist(rng)];
    if (id1 == id2) continue;

    // 같은 pair가 이미 있으면 건너뜀 (원하면 덮어쓰기 해도 됨)
    const image_pair_t pair_id = ImagePair::ImagePairToPairId(id1, id2);
    if (view_graph.image_pairs.count(pair_id) > 0) {
      continue;
    }

    // 2) 완전 "틀린" relative pose 만들기
    ImagePair pair(id1, id2);  // 기본 pose는 identity

    // 랜덤한 ~60 deg 회전
    Eigen::Vector3d axis = Eigen::Vector3d::Random().normalized();
    double angle_rad = 60.0 * M_PI / 180.0;
    Eigen::AngleAxisd aa(angle_rad, axis);
    Eigen::Quaterniond q(aa);

    // 랜덤한 unit translation
    Eigen::Vector3d t = Eigen::Vector3d::Random().normalized();

    pair.cam2_from_cam1 = Rigid3d(q, t);

    // 이 edge는 "그럴듯한" 것처럼 보이게 weight 조금 줌
    pair.is_valid = true;
    pair.weight = 1.0;
    // config, E/F/H, matches, inliers 는 굳이 안 채워도
    // Rotation Averaging / Global Positioning 입장에선
    // 그냥 잘못된 (R,t) constraint 하나가 추가되는 효과만 있다.

    view_graph.image_pairs.emplace(pair_id, std::move(pair));
    view_graph.num_pairs++;
    inserted++;
  }

  // adjacency list 갱신 (이후 단계에서 쓸 수 있으므로)
  view_graph.EstablishAdjacencyList();

  LOG(INFO) << "Injected " << inserted << " synthetic outlier edges into view_graph.";
}

bool GlobalMapper::Solve(const colmap::Database& database,
                         ViewGraph& view_graph,
                         std::unordered_map<camera_t, Camera>& cameras,
                         std::unordered_map<image_t, Image>& images,
                         std::unordered_map<track_t, Track>& tracks) {

  std::vector<image_t> orphan_images;

  // 0. Preprocessing
  if (!options_.skip_preprocessing) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running preprocessing ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();
    // If camera intrinsics seem to be good, force the pair to use essential
    // matrix
    ViewGraphManipulater::UpdateImagePairsConfig(view_graph, cameras, images);
    ViewGraphManipulater::DecomposeRelPose(view_graph, cameras, images);
    run_timer.PrintSeconds();
  }

  // 1. Run view graph calibration
  if (!options_.skip_view_graph_calibration) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running view graph calibration ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    ViewGraphCalibrator vgcalib_engine(options_.opt_vgcalib);
    if (!vgcalib_engine.Solve(view_graph, cameras, images)) {
      return false;
    }
  }

  // 2. Run relative pose estimation
  if (!options_.skip_relative_pose_estimation) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running relative pose estimation ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();
    // Relative pose relies on the undistorted images
    UndistortImages(cameras, images, true);
    EstimateRelativePoses(view_graph, cameras, images, options_.opt_relpose);

    InlierThresholdOptions inlier_thresholds = options_.inlier_thresholds;
    // Undistort the images and filter edges by inlier number
    ImagePairsInlierCount(view_graph, cameras, images, inlier_thresholds, true);

    RelPoseFilter::FilterInlierNum(view_graph,
                                   options_.inlier_thresholds.min_inlier_num);
    RelPoseFilter::FilterInlierRatio(
        view_graph, options_.inlier_thresholds.min_inlier_ratio);

    if (view_graph.KeepLargestConnectedComponents(images) == 0) {
      LOG(ERROR) << "no connected components are found";
      return false;
    }

    run_timer.PrintSeconds();
  }

  // 3. Run rotation averaging for three times
  if (!options_.skip_rotation_averaging) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running rotation averaging ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();

    RotationEstimator ra_engine(options_.opt_ra);
    // The first run is for filtering
    ra_engine.EstimateRotations(view_graph, images);

    RelPoseFilter::FilterRotations(
        view_graph, images, options_.inlier_thresholds.max_rotation_error);
    if (view_graph.KeepLargestConnectedComponents(images) == 0) {
      LOG(ERROR) << "no connected components are found";
      return false;
    }  
    run_timer.PrintSeconds();

    // The second run is for final estimation
    if (!ra_engine.EstimateRotations(view_graph, images)) {
      return false;
    }
    RelPoseFilter::FilterRotations(
        view_graph, images, options_.inlier_thresholds.max_rotation_error);
    image_t num_img = view_graph.KeepLargestConnectedComponents(images);
    if (num_img == 0) {
      LOG(ERROR) << "no connected components are found";
      return false;
    }
    LOG(INFO) << num_img << " / " << images.size()
              << " images are within the connected component." << std::endl;

    run_timer.PrintSeconds();

    // if (options_.num_outlier_edges > 0) {
    //   InjectOutlierEdges(view_graph, images, options_.num_outlier_edges);
    // }
    if (!options_.debug_output_dir.empty()) {
      std::filesystem::path out_path =
          std::filesystem::path(options_.debug_output_dir) /
          "corrupted_view_graph.csv";
      view_graph.DumpCsv(out_path.string());
    }
  }

  // 4. Track establishment and selection
  if (!options_.skip_track_establishment) {
    colmap::Timer run_timer;
    run_timer.Start();

    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running track establishment ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    TrackEngine track_engine(view_graph, images, options_.opt_track);
    std::unordered_map<track_t, Track> tracks_full;
    track_engine.EstablishFullTracks(tracks_full);

    // Filter the tracks
    track_t num_tracks = track_engine.FindTracksForProblem(tracks_full, tracks);
    LOG(INFO) << "Before filtering: " << tracks_full.size()
              << ", after filtering: " << num_tracks << std::endl;

    run_timer.PrintSeconds();
  }

  // 5. Global positioning
  if (!options_.skip_global_positioning) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running global positioning ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();
    // Undistort images in case all previous steps are skipped
    // Skip images where an undistortion already been done
    UndistortImages(cameras, images, false);

    GlobalPositioner gp_engine(options_.opt_gp);
    if (!gp_engine.Solve(view_graph, cameras, images, tracks)) {
      return false;
    }
    orphan_images = gp_engine.OrphanImages();
    LOG(INFO) << orphan_images.size() << " orphan images are found.";

    // If only camera-to-camera constraints are used for solving camera
    // positions, then points needs to be estimated separately
    if (options_.opt_gp.constraint_type ==
        GlobalPositionerOptions::ConstraintType::ONLY_CAMERAS) {
      GlobalPositionerOptions opt_gp_pt = options_.opt_gp;
      opt_gp_pt.constraint_type =
          GlobalPositionerOptions::ConstraintType::ONLY_POINTS;
      opt_gp_pt.optimize_positions = false;
      GlobalPositioner gp_engine_pt(opt_gp_pt);
      if (!gp_engine_pt.Solve(view_graph, cameras, images, tracks)) {
        return false;
      }
    }

    // Filter tracks based on the estimation
    TrackFilter::FilterTracksByAngle(
        view_graph,
        cameras,
        images,
        tracks,
        options_.inlier_thresholds.max_angle_error);
    run_timer.PrintSeconds();
  }

  // 6. Bundle adjustment
  if (!options_.skip_bundle_adjustment) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running bundle adjustment ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    LOG(INFO) << "Bundle adjustment start" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();

    for (int ite = 0; ite < options_.num_iteration_bundle_adjustment; ite++) {
      BundleAdjuster ba_engine(options_.opt_ba);

      BundleAdjusterOptions& ba_engine_options_inner = ba_engine.GetOptions();

      // Staged bundle adjustment
      // 6.1. First stage: optimize positions only
      ba_engine_options_inner.optimize_rotations = false;
      if (!ba_engine.Solve(view_graph, cameras, images, tracks)) {
        return false;
      }
      LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
                << options_.num_iteration_bundle_adjustment
                << ", stage 1 finished (position only)";
      run_timer.PrintSeconds();

      // 6.2. Second stage: optimize rotations if desired
      ba_engine_options_inner.optimize_rotations =
          options_.opt_ba.optimize_rotations;
      if (ba_engine_options_inner.optimize_rotations &&
          !ba_engine.Solve(view_graph, cameras, images, tracks)) {
        return false;
      }
      LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
                << options_.num_iteration_bundle_adjustment
                << ", stage 2 finished";
      if (ite != options_.num_iteration_bundle_adjustment - 1)
        run_timer.PrintSeconds();

      // 6.3. Filter tracks based on the estimation
      // For the filtering, in each round, the criteria for outlier is
      // tightened. If only few tracks are changed, no need to start bundle
      // adjustment right away. Instead, use a more strict criteria to filter
      UndistortImages(cameras, images, true);
      LOG(INFO) << "Filtering tracks by reprojection ...";

      bool status = true;
      size_t filtered_num = 0;
      while (status && ite < options_.num_iteration_bundle_adjustment) {
        double scaling = std::max(3 - ite, 1);
        filtered_num += TrackFilter::FilterTracksByReprojection(
            view_graph,
            cameras,
            images,
            tracks,
            scaling * options_.inlier_thresholds.max_reprojection_error);

        if (filtered_num > 1e-3 * tracks.size()) {
          status = false;
        } else
          ite++;
      }
      if (status) {
        LOG(INFO) << "fewer than 0.1% tracks are filtered, stop the iteration.";
        break;
      }
    }

    // Filter tracks based on the estimation
    UndistortImages(cameras, images, true);
    LOG(INFO) << "Filtering tracks by reprojection ...";
    TrackFilter::FilterTracksByReprojection(
        view_graph,
        cameras,
        images,
        tracks,
        options_.inlier_thresholds.max_reprojection_error);
    TrackFilter::FilterTrackTriangulationAngle(
        view_graph,
        images,
        tracks,
        options_.inlier_thresholds.min_triangulation_angle);

    run_timer.PrintSeconds();
  }

  // 6.5. PnP for orphan images
  if (!options_.skip_global_pnp && 
      options_.opt_gp.center_init_mode ==
        GlobalPositionerOptions::CenterInitMode::SCALED_TRIPLET) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running PnP adjustment ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    LOG(INFO) << "PnP start" << std::endl;

    GlobalPnPOptions pnp_options = options_.opt_pnp;
    GlobalPnP global_pnp(pnp_options);

    if (!orphan_images.empty()) {
      LOG(INFO) << "Running PnP for " << orphan_images.size()
                << " orphan images...";
      global_pnp.Solve(view_graph, orphan_images, cameras, images, tracks);

      BundleAdjuster ba_engine(options_.opt_ba);
      ba_engine.Solve(view_graph, cameras, images, tracks);
    }
  }

  // 7. Retriangulation
  if (!options_.skip_retriangulation) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running retriangulation ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    for (int ite = 0; ite < options_.num_iteration_retriangulation; ite++) {
      colmap::Timer run_timer;
      run_timer.Start();
      RetriangulateTracks(
          options_.opt_triangulator, database, cameras, images, tracks);
      run_timer.PrintSeconds();

      std::cout << "-------------------------------------" << std::endl;
      std::cout << "Running bundle adjustment ..." << std::endl;
      std::cout << "-------------------------------------" << std::endl;
      LOG(INFO) << "Bundle adjustment start" << std::endl;
      BundleAdjuster ba_engine(options_.opt_ba);
      if (!ba_engine.Solve(view_graph, cameras, images, tracks)) {
        return false;
      }

      // Filter tracks based on the estimation
      UndistortImages(cameras, images, true);
      LOG(INFO) << "Filtering tracks by reprojection ...";
      TrackFilter::FilterTracksByReprojection(
          view_graph,
          cameras,
          images,
          tracks,
          options_.inlier_thresholds.max_reprojection_error);
      if (!ba_engine.Solve(view_graph, cameras, images, tracks)) {
        return false;
      }
      run_timer.PrintSeconds();
    }

    // Filter tracks based on the estimation
    UndistortImages(cameras, images, true);
    LOG(INFO) << "Filtering tracks by reprojection ...";
    TrackFilter::FilterTracksByReprojection(
        view_graph,
        cameras,
        images,
        tracks,
        options_.inlier_thresholds.max_reprojection_error);
    TrackFilter::FilterTrackTriangulationAngle(
        view_graph,
        images,
        tracks,
        options_.inlier_thresholds.min_triangulation_angle);
  }

  // 8. Reconstruction pruning
  if (!options_.skip_pruning) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running postprocessing ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();

    // Prune weakly connected images
    PruneWeaklyConnectedImages(images, tracks);

    run_timer.PrintSeconds();
  }

  return true;
}

}  // namespace glomap