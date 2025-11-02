#include "glomap/controllers/global_mapper.h"

#include "glomap/controllers/option_manager.h"
#include "glomap/io/colmap_io.h"
#include "glomap/types.h"

#include <colmap/util/misc.h>
#include <colmap/util/timer.h>

namespace glomap {
// -------------------------------------
// Mappers starting from COLMAP database
// -------------------------------------
int RunMapper(int argc, char** argv) {
  std::string database_path;
  std::string output_path;

  std::string image_path = "";
  // std::string constraint_type = "ONLY_POINTS";
  // std::string center_init_mode = "RANDOM";
  std::string output_format = "bin";

  OptionManager options;
  options.AddRequiredOption("database_path", &database_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_path", &image_path);
  // options.AddDefaultOption("constraint_type",
  //                          &constraint_type,
  //                          "{ONLY_POINTS, ONLY_CAMERAS, "
  //                          "POINTS_AND_CAMERAS_BALANCED, POINTS_AND_CAMERAS, ONLY_SCALEDPOINTS}");
  // options.AddDefaultOption("center_init_mode",
  //                          &center_init_mode,
  //                          "{RANDOM, SCALED_FROM_S, S_ALPHA_LSQ}");
  options.AddDefaultOption("output_format", &output_format, "{bin, txt}");
  options.AddGlobalMapperFullOptions();

  options.Parse(argc, argv);

  if (!colmap::ExistsFile(database_path)) {
    LOG(ERROR) << "`database_path` is not a file";
    return EXIT_FAILURE;
  }

  if (constraint_type == "ONLY_POINTS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::ONLY_POINTS;
  } else if (constraint_type == "ONLY_CAMERAS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::ONLY_CAMERAS;
  } else if (constraint_type == "POINTS_AND_CAMERAS_BALANCED") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::POINTS_AND_CAMERAS_BALANCED;
  } else if (constraint_type == "POINTS_AND_CAMERAS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::POINTS_AND_CAMERAS;
  } else if (constraint_type == "ONLY_SCALEDPOINTS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::ONLY_SCALEDPOINTS;
  } else {
    LOG(ERROR) << "Invalid constriant type";
    return EXIT_FAILURE;
  }

  // // center_init_mode: string -> enum (enum class)
  // using CIM = GlobalPositionerOptions::CenterInitMode;
  // if      (center_init_mode == "RANDOM")        {
  //   options.mapper->opt_gp.center_init_mode = CIM::RANDOM;
  // } else if (center_init_mode == "SCALED_FROM_S") {
  //   options.mapper->opt_gp.center_init_mode = CIM::SCALED_FROM_S;
  // } else if (center_init_mode == "S_ALPHA_LSQ")  {
  //   options.mapper->opt_gp.center_init_mode = CIM::S_ALPHA_LSQ;
  // } else {
  //   LOG(ERROR) << "Invalid center_init_mode";
  //   return EXIT_FAILURE;
  // }

  // Map GlobalPositioning.constraint_type (CLI int) -> enum
  switch (options.mapper->opt_gp.constraint_type_cli) {
    case 0: options.mapper->opt_gp.constraint_type = GlobalPositionerOptions::ONLY_POINTS; break;
    case 1: options.mapper->opt_gp.constraint_type = GlobalPositionerOptions::ONLY_CAMERAS; break;
    case 2: options.mapper->opt_gp.constraint_type = GlobalPositionerOptions::POINTS_AND_CAMERAS_BALANCED; break;
    case 3: options.mapper->opt_gp.constraint_type = GlobalPositionerOptions::POINTS_AND_CAMERAS; break;
    default:
      LOG(ERROR) << "Invalid GlobalPositioning.constraint_type (expected 0~4)";
      return EXIT_FAILURE;
  }

  // // Map GlobalPositioning.center_init_mode (CLI int) -> enum
  // using CIM = GlobalPositionerOptions::CenterInitMode;
  // switch (options.mapper->opt_gp.center_init_mode_cli) {
  //   case 0: options.mapper->opt_gp.center_init_mode = CIM::RANDOM; break;
  //   case 1: options.mapper->opt_gp.center_init_mode = CIM::SCALED_FROM_S; break;
  //   case 2: options.mapper->opt_gp.center_init_mode = CIM::S_ALPHA_LSQ; break;
  //   case 3: options.mapper->opt_gp.center_init_mode = CIM::TRI_RANSAC; break;
  //   case 4: options.mapper->opt_gp.center_init_mode = CIM::LOAD_FROM_CSV; break;
  //   default:
  //     LOG(ERROR) << "Invalid GlobalPositioning.center_init_mode (expected 0~3)";
  //     return EXIT_FAILURE;
  // }

  // using ECM = GlobalPositionerOptions::EdgeConsensusMetric;
  // switch (options.mapper->opt_gp.edge_consensus_cli) {
  //   case 0: options.mapper->opt_gp.edge_consensus_metric = ECM::ANGULAR; break;
  //   case 1: options.mapper->opt_gp.edge_consensus_metric = ECM::PIXEL_REPROJ; break;
  //   default:
  //     LOG(ERROR) << "Invalid gp.edge_consensus (expected 0~1)";
  //     return EXIT_FAILURE;
  // }

  // // Map GlobalPositionerOptions.tri_consensus_cli -> enum
  // using TCM = GlobalPositionerOptions::TriConsensusMetric;
  // switch (options.mapper->opt_gp.tri_consensus_cli) {
  //   case 0: options.mapper->opt_gp.tri_consensus_metric = TCM::ANGULAR; break;
  //   case 1: options.mapper->opt_gp.tri_consensus_metric = TCM::PIXEL_REPROJ; break;
  //   case 2: options.mapper->opt_gp.tri_consensus_metric = TCM::SAMPSON; break;
  //   default:
  //     LOG(ERROR) << "Invalid gp.tri_consensus (expected 0~2)";
  //     return EXIT_FAILURE;
  // }

  // Check whether output_format is valid
  if (output_format != "bin" && output_format != "txt") {
    LOG(ERROR) << "Invalid output format";
    return EXIT_FAILURE;
  }

  // Load the database
  ViewGraph view_graph;
  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;

  const colmap::Database database(database_path);
  ConvertDatabaseToGlomap(database, view_graph, cameras, images);

  if (view_graph.image_pairs.empty()) {
    LOG(ERROR) << "Can't continue without image pairs";
    return EXIT_FAILURE;
  }

  GlobalMapper global_mapper(*options.mapper);

  // Main solver
  LOG(INFO) << "Loaded database";
  colmap::Timer run_timer;
  run_timer.Start();
  global_mapper.Solve(database, view_graph, cameras, images, tracks);
  run_timer.Pause();

  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  WriteGlomapReconstruction(
      output_path, cameras, images, tracks, output_format, image_path);
  LOG(INFO) << "Export to COLMAP reconstruction done";

  return EXIT_SUCCESS;
}

// -------------------------------------
// Mappers starting from COLMAP reconstruction
// -------------------------------------
int RunMapperResume(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string image_path = "";
  std::string output_format = "bin";

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_path", &image_path);
  options.AddDefaultOption("output_format", &output_format, "{bin, txt}");
  options.AddGlobalMapperResumeFullOptions();

  options.Parse(argc, argv);

  if (!colmap::ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  // Check whether output_format is valid
  if (output_format != "bin" && output_format != "txt") {
    LOG(ERROR) << "Invalid output format";
    return EXIT_FAILURE;
  }

  // Load the reconstruction
  ViewGraph view_graph;       // dummy variable
  colmap::Database database;  // dummy variable

  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;
  colmap::Reconstruction reconstruction;
  reconstruction.Read(input_path);
  ConvertColmapToGlomap(reconstruction, cameras, images, tracks);

  GlobalMapper global_mapper(*options.mapper);

  // Main solver
  colmap::Timer run_timer;
  run_timer.Start();
  global_mapper.Solve(database, view_graph, cameras, images, tracks);
  run_timer.Pause();

  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  WriteGlomapReconstruction(
      output_path, cameras, images, tracks, output_format, image_path);
  LOG(INFO) << "Export to COLMAP reconstruction done";

  return EXIT_SUCCESS;
}

}  // namespace glomap