// World-model experiment (replication dossier, finding 22): the dissertation's own
// quantitative test of its central claim — thesis §9.2, Abb. 9.1/9.2; Backer &
// Mertsching, WAPCV 2003, Fig. 4/5.
//
// A scene of small objects, some static, some moving, is explored by two
// attention models that see the same simulated master map of attention (no
// features). Each has to keep a world model: which object is where. Object
// recognition is simulated — it names the object under the focus correctly, at a
// fixed cost in frames.
//
//   conventional  after Koch & Ullman: blur, subtract a decaying inhibition map,
//                 take the maximum; recognition takes 3 frames; the identity is
//                 bound to the location where the object was selected; an 8 x 8
//                 area is then inhibited, decaying by 20% per frame
//   two-stage     one 2D neural field of local inhibition -> activity clusters ->
//                 object files -> behaviour "exploration"; recognition takes 4
//                 frames (one extra, to charge for the field); the identity is
//                 bound to the object file and moves with its cluster
//
// Measures, averaged over all frames of a run: the number of objects whose
// identity the world model holds at a believed position within 20 px of the true
// one ("recognized"), and the position error of those.
//
// What the sources fix: object size 5 x 5, spacing >= 14 px at every moment,
// motion <= 2 px per axis and frame on a straight path, uniform noise of half the
// object amplitude, 40 frames, 50 runs per condition, the conventional model's
// parameters, the 20-px criterion, 3 vs 4 frames. What they leave open, and what
// is chosen here (each a command-line option): the map size (128), the blur of
// the conventional model (Gaussian, sigma 1.5), the inhibition strength (1), the
// field's input gain and update cycles per frame (the dissertation system's:
// 0.765 and a fixed 20), the distribution of velocities (uniform in [-2, 2] per
// axis, at least 0.5 px per frame), the noise's mean (zero), the
// correspondence radius (5 px, the field's local-maximum range x_a as measured
// in the dossier, finding 21).
//
// Output: one JSON document on stdout (read by eval/replication.py).

#include "attention/selection/neural_field_selection.h"
#include "attention/system/behavior.h"
#include "attention/system/object_file.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using attention::selection::NeuralFieldSelection;
using attention::selection::SelectionParams;
namespace sys = attention::system;

namespace
{

struct Options
{
  int size = 128;
  int frames = 40;
  int runs = 50;
  int seed0 = 0;
  float noise = 0.5f;        // uniform noise amplitude (peak to peak), object amplitude 1
  bool zero_mean = true;     // noise in [-noise/2, noise/2] (the thesis's field experiments, dossier
                             // finding 18) or in [0, noise]
  float blur_sigma = 1.5f;   // conventional model
  float inhibition = 1.0f;   // conventional model: strength of a fresh mark
  int conventional_cost = 3; // frames per recognition
  // The conventional model's recognition names the object under the focus. The
  // sources do not say when it looks: at selection (default — the reading that
  // favours the conventional model) or when recognition ends, three frames
  // later, by which time a moving object may have left the focus.
  bool late_identity = false;
  int two_stage_cost = 4;
  // The dissertation system ran its field for a fixed 20 cycles per frame: its
  // convergence test compared the *summed* |du| with 0.002, which a field with
  // noise on its input never meets (nf2d.h, readargs.h: updateCycles = 20).
  int cycles = 20;      // field update cycles per frame, fixed; 0 = to convergence, at most max_cycles
  int max_cycles = 55;  // thesis ch. 6: 55; the dissertation system's default per frame: 20
  float thresh = 0.02f; // convergence: mean |du| (thesis Abb. 6.8: 0.02; the dissertation system: 0.002)
  float gain = 0.765f;  // field input gain: the dissertation system's 0.003 on a 0-255 master map
  double radius = 5.0;
  std::string params = "esab2"; // field parameter set: esab2 | port
  std::vector<int> statics = {1, 3, 5};
  std::vector<int> dynamics = {0, 1, 2, 3, 4, 5};
};

constexpr int kObject = 5;           // object side (px)
constexpr double kSpacing = 14.0;    // minimal centre distance at every moment
constexpr int kMargin = 14;          // objects stay this far from the map's border (field border term: 9)
constexpr double kRecognized = 20.0; // px: beyond this the object does not count as recognized

// std::uniform_real_distribution is implementation-defined: libc++ and
// libstdc++ draw different sequences from the same seeded engine, so the same
// seed gave different scenes on macOS and Linux (found by the cross-platform
// check, .github/workflows/replication.yml). This sampler is portable.
double uniform(std::mt19937& rng, double lo, double hi)
{
  const double u =
      (rng() - std::mt19937::min()) / (static_cast<double>(std::mt19937::max()) - std::mt19937::min() + 1.0);
  return lo + u * (hi - lo);
}

struct SceneObject
{
  cv::Point2d start;
  cv::Point2d velocity;
  // Where the object is drawn — whole pixels; this is also the truth that the
  // world models are scored against.
  cv::Point2d at(int frame) const
  {
    const cv::Point2d p = start + velocity * static_cast<double>(frame);
    return cv::Point2d(std::round(p.x), std::round(p.y));
  }
};

// Rejection sampling, one object at a time: straight paths that stay inside the
// map and keep their distance from every object placed before, for the whole run.
std::vector<SceneObject> make_scene(int n_static, int n_dynamic, const Options& opt, std::mt19937& rng)
{
  auto position = [&](std::mt19937& rng) { return uniform(rng, kMargin, opt.size - kMargin); };
  auto step = [&](std::mt19937& rng) { return uniform(rng, -2.0, 2.0); };
  auto fits = [&](const SceneObject& candidate, const std::vector<SceneObject>& placed)
  {
    for (int t = 0; t < opt.frames; ++t)
    {
      const cv::Point2d p = candidate.at(t);
      if (p.x < kMargin || p.y < kMargin || p.x > opt.size - kMargin || p.y > opt.size - kMargin)
      {
        return false;
      }
      for (const auto& other : placed)
      {
        if (cv::norm(p - other.at(t)) < kSpacing)
        {
          return false;
        }
      }
    }
    return true;
  };
  for (int restart = 0; restart < 200; ++restart)
  {
    std::vector<SceneObject> objects;
    // Dynamic objects first (they are the hard ones to place); the order in the
    // result is static, then dynamic
    std::vector<SceneObject> dynamic, fixed;
    bool ok = true;
    for (int i = 0; i < n_static + n_dynamic && ok; ++i)
    {
      const bool moving = i < n_dynamic;
      ok = false;
      for (int attempt = 0; attempt < 5000 && !ok; ++attempt)
      {
        SceneObject object;
        object.start = cv::Point2d(std::floor(position(rng)), std::floor(position(rng)));
        if (moving)
        {
          do
          {
            object.velocity = cv::Point2d(step(rng), step(rng));
          } while (cv::norm(object.velocity) < 0.5); // a dynamic object does move
        }
        if (fits(object, objects))
        {
          objects.push_back(object);
          (moving ? dynamic : fixed).push_back(object);
          ok = true;
        }
      }
    }
    if (ok)
    {
      fixed.insert(fixed.end(), dynamic.begin(), dynamic.end());
      return fixed;
    }
  }
  throw std::runtime_error("world_model: no valid scene found (map too small for the object count?)");
}

cv::Rect object_rect(const cv::Point2d& centre)
{
  return cv::Rect(static_cast<int>(centre.x) - kObject / 2, static_cast<int>(centre.y) - kObject / 2, kObject, kObject);
}

cv::Mat render(const std::vector<SceneObject>& objects, int frame, const Options& opt, std::mt19937& rng)
{
  cv::Mat map(opt.size, opt.size, CV_32F);
  const double low = opt.zero_mean ? -0.5 * opt.noise : 0.0;
  for (int y = 0; y < map.rows; ++y)
  {
    float* row = map.ptr<float>(y);
    for (int x = 0; x < map.cols; ++x)
    {
      row[x] = static_cast<float>(uniform(rng, low, low + opt.noise));
    }
  }
  for (const auto& object : objects)
  {
    map(object_rect(object.at(frame))) += 1.0f;
  }
  return map;
}

// The simulated recognition: the object at this position (its 5 x 5 area, one
// pixel of slack for a centroid that sits on the edge), or -1.
int object_at(const std::vector<SceneObject>& objects, int frame, const cv::Point2d& where)
{
  int best = -1;
  double best_distance = kObject / 2.0 + 1.5;
  for (size_t i = 0; i < objects.size(); ++i)
  {
    const cv::Point2d p = objects[i].at(frame);
    const double d = std::max(std::abs(p.x - where.x), std::abs(p.y - where.y));
    if (d < best_distance)
    {
      best_distance = d;
      best = static_cast<int>(i);
    }
  }
  return best;
}

struct Score
{
  double recognized = 0.0; // mean over frames
  double error_sum = 0.0;
  long error_count = 0;
  double clusters = 0.0; // two-stage only: mean activity clusters per frame
  double position_error() const { return error_count > 0 ? error_sum / error_count : 0.0; }
};

// believed: identity -> believed position. Accumulates one frame into the score.
void score_frame(const std::map<int, cv::Point2d>& believed, const std::vector<SceneObject>& objects, int frame,
                 int frames, Score& score)
{
  for (const auto& entry : believed)
  {
    const double error = cv::norm(entry.second - objects[entry.first].at(frame));
    if (error <= kRecognized)
    {
      score.recognized += 1.0 / frames;
      score.error_sum += error;
      score.error_count += 1;
    }
  }
}

Score run_conventional(const std::vector<SceneObject>& objects, const std::vector<cv::Mat>& maps, const Options& opt)
{
  Score score;
  cv::Mat inhibition = cv::Mat::zeros(opt.size, opt.size, CV_32F);
  std::map<int, cv::Point2d> believed;
  bool busy = false;
  int busy_since = 0;
  int identity = -1;
  cv::Point focus;
  for (int t = 0; t < opt.frames; ++t)
  {
    inhibition *= 0.8f;
    if (!busy)
    {
      cv::Mat blurred;
      cv::GaussianBlur(maps[t], blurred, cv::Size(0, 0), opt.blur_sigma);
      blurred -= inhibition;
      cv::minMaxLoc(blurred, nullptr, nullptr, nullptr, &focus);
      identity = object_at(objects, t, focus);
      busy = true;
      busy_since = t;
    }
    if (busy && t - busy_since + 1 >= opt.conventional_cost)
    {
      // Recognition done: the identity goes to the place where it was selected
      // — there is nothing else to bind it to — and the place is inhibited.
      if (opt.late_identity)
      {
        identity = object_at(objects, t, focus);
      }
      if (identity >= 0)
      {
        believed[identity] = cv::Point2d(focus.x, focus.y);
      }
      const cv::Rect mark = cv::Rect(focus.x - 4, focus.y - 4, 8, 8) & cv::Rect(0, 0, opt.size, opt.size);
      inhibition(mark).setTo(opt.inhibition);
      busy = false;
    }
    score_frame(believed, objects, t, opt.frames, score);
  }
  return score;
}

NeuralFieldSelection make_field(const Options& opt)
{
  NeuralFieldSelection::Params params;
  if (opt.params == "esab2")
  {
    // What the dissertation system configured for its single 2D field (dossier,
    // finding B) — configs/thesis/thesis.yaml
    params.alpha = 0.33f;
    params.global_mult = 8.0f;
    params.resting = -0.33f;
    params.kernel_size = 15;
    params.kernel_s = 3.3f;
    params.kernel_k = 0.12f;
    params.kernel_s2 = 14.0f;
    params.kernel_k2 = 0.03f;
  }
  params.max_cycles = opt.max_cycles;
  params.change_thresh = opt.thresh;
  params.cycles_per_frame = opt.cycles;
  params.input_mult = opt.gain;
  params.field_max_size = opt.size; // the field at the map's resolution
  return NeuralFieldSelection(SelectionParams{}, params);
}

Score run_two_stage(const std::vector<SceneObject>& objects, const std::vector<cv::Mat>& maps, const Options& opt)
{
  Score score;
  const NeuralFieldSelection field = make_field(opt);
  cv::Mat activity;

  sys::ObjectFileStore::Config store_config;
  store_config.rule = sys::ObjectFileStore::Rule::Thesis; // no feature maps here: position decides
  store_config.correspondence_radius = opt.radius;
  sys::ObjectFileStore store(store_config);
  sys::Exploration::Params behaviour_params;
  behaviour_params.dwell_frames = opt.two_stage_cost;
  sys::Exploration behaviour(behaviour_params);

  std::map<int, int> identity_of; // object-file label -> recognized identity
  int held_label = -1;
  int held_frames = 0;
  for (int t = 0; t < opt.frames; ++t)
  {
    std::vector<sys::Cluster> clusters;
    for (const auto& active : field.track(maps[t], activity))
    {
      sys::Cluster cluster;
      cluster.centroid =
          cv::Point(static_cast<int>(active.centroid.x + 0.5f), static_cast<int>(active.centroid.y + 0.5f));
      cluster.centroid_exact = active.centroid;
      cluster.bbox = active.bbox;
      cluster.size = active.size;
      cluster.mean_saliency = active.mean_input;
      clusters.push_back(cluster);
    }
    score.clusters += static_cast<double>(clusters.size()) / opt.frames;
    store.update(clusters, t);

    const sys::ObjectFile* focus = behaviour.select_focus(store, t);
    if (focus == nullptr)
    {
      held_label = -1;
      held_frames = 0;
    }
    else
    {
      held_frames = focus->label == held_label ? held_frames + 1 : 1;
      held_label = focus->label;
      if (held_frames == opt.two_stage_cost)
      {
        const int identity = object_at(objects, t, cv::Point2d(focus->centroid_exact.x, focus->centroid_exact.y));
        if (identity >= 0)
        {
          identity_of[focus->label] = identity;
        }
        held_frames = 0; // a focus that stays is a new recognition
      }
    }

    // The world model: every recognized file, at its cluster's position (an
    // inactive file: where it was last). If two files carry one identity, the
    // one seen last speaks for it.
    std::map<int, cv::Point2d> believed;
    std::map<int, int> seen;
    auto consider = [&](const sys::ObjectFile& file)
    {
      const auto it = identity_of.find(file.label);
      if (it == identity_of.end())
      {
        return;
      }
      if (seen.count(it->second) == 0 || file.last_seen_frame > seen[it->second])
      {
        seen[it->second] = file.last_seen_frame;
        believed[it->second] = cv::Point2d(file.centroid_exact.x, file.centroid_exact.y);
      }
    };
    for (const auto& file : store.inactive_files())
    {
      consider(file);
    }
    for (const auto& file : store.active_files())
    {
      consider(file);
    }
    if (std::getenv("WORLD_MODEL_TRACE") != nullptr)
    {
      std::cerr << "t=" << t << " clusters=" << clusters.size() << " active=" << store.active_files().size()
                << " inactive=" << store.inactive_files().size() << " focus=" << held_label << " |";
      for (const auto& entry : believed)
      {
        std::cerr << " " << entry.first << ":" << std::fixed << std::setprecision(1)
                  << cv::norm(entry.second - objects[entry.first].at(t));
      }
      std::cerr << "\n";
    }
    score_frame(believed, objects, t, opt.frames, score);
  }
  return score;
}

std::string num(double v)
{
  std::ostringstream s;
  s << std::setprecision(6) << v;
  return s.str();
}

std::vector<int> parse_list(const std::string& text)
{
  std::vector<int> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ','))
  {
    values.push_back(std::stoi(item));
  }
  return values;
}

void print_help()
{
  std::cout << "Usage: world_model [options]\n\n"
               "The dissertation's world-model experiment (thesis 9.2, Abb. 9.2; WAPCV 2003, Fig. 5):\n"
               "a conventional attention model with a static inhibition map against the two-stage\n"
               "model (neural field -> object files -> exploration) on simulated master maps.\n"
               "Prints one JSON document with per-run scores for every condition.\n\n"
               "  --runs N            runs per condition (default 50, as in the thesis)\n"
               "  --seed0 N           first seed; run i uses seed0 + i (default 0)\n"
               "  --static A,B,..     numbers of static objects (default 1,3,5)\n"
               "  --dynamic A,B,..    numbers of dynamic objects (default 0,1,2,3,4,5)\n"
               "  --frames N          frames per run (default 40)\n"
               "  --size N            side of the master map in px (default 128)\n"
               "  --noise A           uniform noise amplitude, object amplitude 1 (default 0.5)\n"
               "  --positive-noise    noise in [0, A] instead of zero-mean [-A/2, A/2]\n"
               "  --params NAME       field parameters: esab2 (the dissertation system's, default) | port\n"
               "  --cycles N          field update cycles per frame (default 20, the dissertation system's); 0 = to "
               "convergence\n"
               "  --max-cycles N      cycle limit when running to convergence (default 55)\n"
               "  --thresh T          convergence threshold, mean |du| (default 0.02; the dissertation system: 0.002)\n"
               "  --gain G            field input gain (default 0.765, the dissertation system's)\n"
               "  --radius R          correspondence radius in px (default 5)\n"
               "  --late-identity     conventional model: recognition names what is under the focus when it\n"
               "                      ends, not when it starts (sensitivity variant)\n"
               "  --blur S            conventional model: Gaussian blur sigma (default 1.5)\n"
               "  --inhibition A      conventional model: strength of a fresh inhibition mark (default 1)\n"
               "  --help              this text\n";
}

} // namespace

int main(int argc, char** argv)
{
  Options opt;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string
    {
      if (i + 1 >= argc)
      {
        throw std::runtime_error("world_model: " + arg + " needs a value");
      }
      return argv[++i];
    };
    try
    {
      if (arg == "--help" || arg == "-h")
      {
        print_help();
        return 0;
      }
      else if (arg == "--runs")
        opt.runs = std::stoi(value());
      else if (arg == "--seed0")
        opt.seed0 = std::stoi(value());
      else if (arg == "--static")
        opt.statics = parse_list(value());
      else if (arg == "--dynamic")
        opt.dynamics = parse_list(value());
      else if (arg == "--frames")
        opt.frames = std::stoi(value());
      else if (arg == "--size")
        opt.size = std::stoi(value());
      else if (arg == "--noise")
        opt.noise = std::stof(value());
      else if (arg == "--late-identity")
        opt.late_identity = true;
      else if (arg == "--positive-noise")
        opt.zero_mean = false;
      else if (arg == "--params")
        opt.params = value();
      else if (arg == "--cycles")
        opt.cycles = std::stoi(value());
      else if (arg == "--max-cycles")
        opt.max_cycles = std::stoi(value());
      else if (arg == "--thresh")
        opt.thresh = std::stof(value());
      else if (arg == "--gain")
        opt.gain = std::stof(value());
      else if (arg == "--radius")
        opt.radius = std::stod(value());
      else if (arg == "--blur")
        opt.blur_sigma = std::stof(value());
      else if (arg == "--inhibition")
        opt.inhibition = std::stof(value());
      else
      {
        std::cerr << "world_model: unknown option " << arg << " (see --help)\n";
        return 2;
      }
    }
    catch (const std::exception& e)
    {
      std::cerr << e.what() << "\n";
      return 2;
    }
  }
  if (opt.params != "esab2" && opt.params != "port")
  {
    std::cerr << "world_model: --params must be esab2 or port\n";
    return 2;
  }

  std::cout << "{\n  \"experiment\": \"world-model\",\n  \"size\": " << opt.size << ", \"frames\": " << opt.frames
            << ", \"runs\": " << opt.runs << ", \"seed0\": " << opt.seed0 << ", \"noise\": " << num(opt.noise)
            << ", \"zero_mean\": " << (opt.zero_mean ? "true" : "false")
            << ", \"late_identity\": " << (opt.late_identity ? "true" : "false") << ", \"params\": \"" << opt.params
            << "\", \"cycles\": " << opt.cycles << ", \"max_cycles\": " << opt.max_cycles
            << ", \"thresh\": " << num(opt.thresh) << ", \"gain\": " << num(opt.gain)
            << ", \"radius\": " << num(opt.radius) << ",\n  \"conditions\": [\n";
  bool first = true;
  for (int n_static : opt.statics)
  {
    for (int n_dynamic : opt.dynamics)
    {
      std::cout << (first ? "" : ",\n") << "    {\"static\": " << n_static << ", \"dynamic\": " << n_dynamic
                << ", \"runs\": [";
      first = false;
      for (int run = 0; run < opt.runs; ++run)
      {
        // One generator for the scene and its noise: both models see the same frames
        std::mt19937 rng(static_cast<unsigned>(opt.seed0 + run) * 7919u + static_cast<unsigned>(n_static) * 101u +
                         static_cast<unsigned>(n_dynamic));
        const std::vector<SceneObject> objects = make_scene(n_static, n_dynamic, opt, rng);
        std::vector<cv::Mat> maps;
        for (int t = 0; t < opt.frames; ++t)
        {
          maps.push_back(render(objects, t, opt, rng));
        }
        const Score conventional = run_conventional(objects, maps, opt);
        const Score two_stage = run_two_stage(objects, maps, opt);
        std::cout << (run ? ", " : "") << "{\"conventional\": [" << num(conventional.recognized) << ", "
                  << num(conventional.position_error()) << "], \"two_stage\": [" << num(two_stage.recognized) << ", "
                  << num(two_stage.position_error()) << "], \"clusters\": " << num(two_stage.clusters) << "}";
      }
      std::cout << "]}";
    }
  }
  std::cout << "\n  ]\n}\n";
  return 0;
}
