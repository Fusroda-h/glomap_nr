#include "glomap/controllers/global_mapper.h"
#include "glomap/controllers/option_manager.h"
#include "glomap/io/colmap_io.h"
#include "glomap/types.h"

#include <colmap/util/misc.h>
#include <colmap/util/timer.h>

namespace glomap {

int RunMapper(int argc, char** argv) {
  std::string database_path;
  std::string output_path;
  std::string image_path = "";
  std::string output_format = "bin";

  OptionManager options;
  options.AddRequiredOption("database_path", &database_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_path", &image_path);
  options.AddDefaultOption("output_format", &output_format, "{bin, txt}");
  // This will add GlobalPositioning.* options, but we will not do extra CLI mapping here.
  options.AddGlobalMapperFullOptions();

  options.Parse(argc, argv);

  if (!colmap::ExistsFile(database_path)) {
    LOG(ERROR) << "`database_path` is not a file";
    return EXIT_FAILURE;
  }

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

  // Map GlobalPositioning.center_init_mode (CLI int) -> enum
  using CIM = GlobalPositionerOptions::CenterInitMode;
  switch (options.mapper->opt_gp.center_init_mode_cli) {
    case 0: options.mapper->opt_gp.center_init_mode = CIM::RANDOM; break;
    case 1: options.mapper->opt_gp.center_init_mode = CIM::SCALED_TRIPLET; break;
    // case 2: options.mapper->opt_gp.center_init_mode = CIM::LOAD_FROM_CSV; break;
    default:
      LOG(ERROR) << "Invalid GlobalPositioning.center_init_mode (expected 0~3)";
      return EXIT_FAILURE;
  }

  // At this point, options.mapper->opt_gp is already filled from CLI / default.
  // We do NOT do extra string-based mapping here anymore.

  if (output_format != "bin" && output_format != "txt") {
    LOG(ERROR) << "Invalid output format";
    return EXIT_FAILURE;
  }

  // Load database
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

  if (output_format != "bin" && output_format != "txt") {
    LOG(ERROR) << "Invalid output format";
    return EXIT_FAILURE;
  }

  ViewGraph view_graph;       // dummy
  colmap::Database database;  // dummy

  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;
  colmap::Reconstruction reconstruction;
  reconstruction.Read(input_path);
  ConvertColmapToGlomap(reconstruction, cameras, images, tracks);

  GlobalMapper global_mapper(*options.mapper);

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
