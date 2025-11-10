#include "glomap/processors/track_filter.h"

#include "glomap/math/rigid3d.h"

namespace glomap {

int TrackFilter::FilterTracksByReprojection(
    const ViewGraph& view_graph,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    double max_reprojection_error,
    bool in_normalized_image) {
  int counter = 0;
  for (auto& [track_id, track] : tracks) {
    std::vector<Observation> observation_new;
    observation_new.reserve(track.observations.size());
    for (auto& [image_id, feature_id] : track.observations) {
      // Check if image exists
      auto it_img = images.find(image_id);
      if (it_img == images.end()) {
        // If the image is missing, keep the observation as-is (for possible PnP use)
        observation_new.emplace_back(image_id, feature_id);
        continue;
      }
      const Image& image = it_img->second;

      // Orphan (unregistered) views are excluded from reprojection filtering but their observations are kept
      if (!image.is_registered) {
        observation_new.emplace_back(image_id, feature_id);
        continue;
      }

      Eigen::Vector3d pt_calc = image.cam_from_world * track.xyz;
      if (pt_calc(2) < EPS) {
        // Invalid depth, ignore this observation
        continue;
      }

      double reprojection_error = max_reprojection_error;
      if (in_normalized_image) {
        const Eigen::Vector3d& feature_undist =
            image.features_undist.at(feature_id);

        Eigen::Vector2d pt_reproj = pt_calc.head(2) / pt_calc(2);
        reprojection_error =
            (pt_reproj - feature_undist.head(2) / (feature_undist(2) + EPS))
                .norm();
      } else {
        Eigen::Vector2d pt_reproj = pt_calc.head(2) / pt_calc(2);
        Eigen::Vector2d pt_dist =
            cameras.at(image.camera_id).ImgFromCam(pt_reproj);
        reprojection_error =
            (pt_dist - image.features.at(feature_id)).norm();
      }

      // Apply reprojection filtering only for registered views
      if (reprojection_error < max_reprojection_error) {
        observation_new.emplace_back(image_id, feature_id);
      }
    }
    if (observation_new.size() != track.observations.size()) {
      counter++;
      track.observations = std::move(observation_new);
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << tracks.size()
            << " tracks by reprojection error";
  return counter;
}

int TrackFilter::FilterTracksByAngle(
    const ViewGraph& view_graph,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    double max_angle_error) {
  int counter = 0;
  double thres = std::cos(DegToRad(max_angle_error));
  double thres_uncalib = std::cos(DegToRad(max_angle_error * 2));
  for (auto& [track_id, track] : tracks) {
    std::vector<Observation> observation_new;
    observation_new.reserve(track.observations.size());
    for (auto& [image_id, feature_id] : track.observations) {
      auto it_img = images.find(image_id);
      if (it_img == images.end()) {
        // If image is missing, keep the observation as-is
        observation_new.emplace_back(image_id, feature_id);
        continue;
      }
      const Image& image = it_img->second;

      // Orphan (unregistered) views are excluded from angle filtering but their observations are kept
      if (!image.is_registered) {
        observation_new.emplace_back(image_id, feature_id);
        continue;
      }

      const Eigen::Vector3d& feature_undist =
          image.features_undist.at(feature_id);
      Eigen::Vector3d pt_calc = image.cam_from_world * track.xyz;
      if (pt_calc(2) < EPS) {
        // Invalid depth, ignore this observation
        continue;
      }

      pt_calc = pt_calc.normalized();
      double thres_cam =
          (cameras.at(image.camera_id).has_prior_focal_length)
              ? thres
              : thres_uncalib;

      // Apply angular filtering only for registered views
      if (pt_calc.dot(feature_undist) > thres_cam) {
        observation_new.emplace_back(image_id, feature_id);
      }
    }
    if (observation_new.size() != track.observations.size()) {
      counter++;
      track.observations = std::move(observation_new);
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << tracks.size()
            << " tracks by angle error";
  return counter;
}

int TrackFilter::FilterTrackTriangulationAngle(
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    double min_angle) {
  int counter = 0;
  double thres = std::cos(DegToRad(min_angle));
  for (auto& [track_id, track] : tracks) {
    // Use only registered cameras when computing triangulation angle
    std::vector<Eigen::Vector3d> pts_calc;
    pts_calc.reserve(track.observations.size());
    for (auto& [image_id, feature_id] : track.observations) {
      auto it_img = images.find(image_id);
      if (it_img == images.end()) continue;

      const Image& image = it_img->second;
      if (!image.is_registered) {
        // Orphan cameras are ignored for baseline computation,
        // but their observations are kept in the track.
        continue;
      }

      Eigen::Vector3d pt_calc = (track.xyz - image.Center()).normalized();
      pts_calc.emplace_back(pt_calc);
    }

    const int n = static_cast<int>(pts_calc.size());
    // If there are fewer than 2 registered views, we cannot evaluate
    // the triangulation angle in a meaningful way. Keep this track
    // as-is so that it can still be used (e.g., for PnP with orphans).
    if (n < 2) {
      continue;
    }

    bool status = false;
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        if (pts_calc[i].dot(pts_calc[j]) < thres) {
          status = true;
          break;
        }
      }
      if (status) break;
    }

    // If no pair of registered cameras has a large enough baseline
    // (angle < min_angle for all pairs), then drop this track.
    if (!status) {
      counter++;
      track.observations.clear();
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << tracks.size()
            << " tracks by too small triangulation angle";
  return counter;
}

}  // namespace glomap