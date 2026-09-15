// M6 tests: the symbolic second stage — object files, the Exploration
// behavior, and the AttentionSystem end to end on the motion sequence.

#include "attention/system/attention_system.h"
#include "attention/system/behavior.h"
#include "attention/system/object_file.h"
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace attention;

namespace
{
system::Cluster cluster_at(int x, int y, float saliency, int size = 100)
{
  system::Cluster c;
  c.centroid = cv::Point(x, y);
  c.bbox = cv::Rect(x - 5, y - 5, 10, 10);
  c.size = size;
  c.mean_saliency = saliency;
  return c;
}

// Label of the active file nearest a point (0 if none).
int label_near(const system::ObjectFileStore& store, cv::Point p, double tol = 8.0)
{
  for (const auto& f : store.active_files())
  {
    if (cv::norm(f.centroid - p) <= tol)
    {
      return f.label;
    }
  }
  return 0;
}
} // namespace

TEST_CASE("object files: creation, tracking, new-object, revival", "[system][objectfile]")
{
  system::ObjectFileStore::Config cfg;
  cfg.correspondence_radius = 30.0;
  cfg.max_inactive_age = 5;
  system::ObjectFileStore store(cfg);

  // Frame 0: two objects created with distinct labels.
  store.update({cluster_at(20, 20, 0.9f), cluster_at(100, 100, 0.5f)}, 0);
  REQUIRE(store.active_files().size() == 2);
  const int a = label_near(store, {20, 20});
  const int b = label_near(store, {100, 100});
  REQUIRE(a != 0);
  REQUIRE(b != 0);
  REQUIRE(a != b);

  // Frame 1: both move slightly -> same labels (tracked, not recreated).
  store.update({cluster_at(24, 22, 0.9f), cluster_at(103, 101, 0.5f)}, 1);
  REQUIRE(store.active_files().size() == 2);
  CHECK(label_near(store, {24, 22}) == a);
  CHECK(label_near(store, {103, 101}) == b);

  // Frame 2: A persists, B disappears -> B goes inactive.
  store.update({cluster_at(26, 23, 0.9f)}, 2);
  CHECK(store.active_files().size() == 1);
  CHECK(label_near(store, {26, 23}) == a);
  CHECK(store.inactive_files().size() == 1);

  // A far new cluster creates a new label (not a match to A or inactive B).
  store.update({cluster_at(28, 24, 0.9f), cluster_at(200, 40, 0.7f)}, 3);
  const int c = label_near(store, {200, 40});
  CHECK(c != 0);
  CHECK(c != a);
  CHECK(c != b);

  // B reappears near its last position -> revived with its original label.
  store.update({cluster_at(30, 25, 0.9f), cluster_at(103, 101, 0.5f)}, 4);
  CHECK(label_near(store, {103, 101}) == b);
}

TEST_CASE("object files: inactive files age out", "[system][objectfile]")
{
  system::ObjectFileStore::Config cfg;
  cfg.max_inactive_age = 3;
  system::ObjectFileStore store(cfg);

  store.update({cluster_at(50, 50, 0.8f)}, 0); // create
  store.update({}, 1);                         // no clusters -> inactive
  REQUIRE(store.inactive_files().size() == 1);

  store.update({}, 2);
  store.update({}, 3);
  CHECK(store.inactive_files().size() == 1); // frame 3 - last_seen 0 = 3, not > 3
  store.update({}, 4);
  CHECK(store.inactive_files().empty()); // now aged out
}

TEST_CASE("Exploration dwells, then switches by inhibition of return", "[system][behavior]")
{
  system::ObjectFileStore store;
  system::Exploration::Params params;
  params.dwell_frames = 3;
  system::Exploration behavior(params);

  const auto strong = cluster_at(20, 20, 0.9f);
  const auto weak = cluster_at(150, 150, 0.5f);

  auto step = [&](int frame)
  {
    store.update({strong, weak}, frame);
    const system::ObjectFile* focus = behavior.select_focus(store, frame);
    return focus ? focus->label : 0;
  };

  const int strong_label = [&]
  {
    store.update({strong, weak}, 0);
    return label_near(store, {20, 20});
  }();
  store.reset();
  behavior.reset();

  // Frames 0-2: dwell on the stronger (and never-selected) object.
  CHECK(step(0) == strong_label);
  CHECK(step(1) == strong_label);
  CHECK(step(2) == strong_label);
  // Frame 3: dwell complete -> switch to the still-never-selected weaker one.
  const int other = step(3);
  CHECK(other != strong_label);
  CHECK(other != 0);
  // Frames 4-5: dwell on the second object.
  CHECK(step(4) == other);
  CHECK(step(5) == other);
  // Frame 6: both selected; the longest-unselected (the first) comes back.
  CHECK(step(6) == strong_label);
}

TEST_CASE("appearance matching keeps object identity through a crossing", "[system][objectfile][appearance]")
{
  // Two objects of different colour swap positions. Position-only correspondence
  // swaps their labels at the crossing; folding appearance into the cost keeps
  // each label with its own colour — the DeepSORT idea using our own features.
  auto make_cluster = [](int x, int y, cv::Vec3f colour)
  {
    system::Cluster c;
    c.centroid = cv::Point(x, y);
    c.bbox = cv::Rect(x - 10, y - 10, 20, 20);
    c.size = 400;
    c.mean_saliency = 0.8f;
    c.appearance = colour;
    return c;
  };
  const cv::Vec3f red(0, 0, 255);
  const cv::Vec3f blue(255, 0, 0);

  auto red_label_after_crossing = [&](bool appearance_matching)
  {
    system::ObjectFileStore::Config cfg;
    cfg.correspondence_radius = 50.0;
    cfg.appearance_matching = appearance_matching;
    system::ObjectFileStore store(cfg);

    store.update({make_cluster(60, 100, red), make_cluster(140, 100, blue)}, 0);
    const int red_label = label_near(store, {60, 100});
    store.update({make_cluster(90, 100, red), make_cluster(110, 100, blue)}, 1); // approaching
    store.update({make_cluster(140, 100, red), make_cluster(60, 100, blue)}, 2); // crossed over
    return std::make_pair(red_label, label_near(store, {140, 100}));             // label now on the red object
  };

  const auto with_appearance = red_label_after_crossing(true);
  CHECK(with_appearance.second == with_appearance.first); // identity preserved

  const auto position_only = red_label_after_crossing(false);
  CHECK(position_only.second != position_only.first); // identity swapped at the crossing
}

namespace
{
system::Cluster coloured_at(int x, int y, const cv::Vec3f& colour)
{
  system::Cluster c = cluster_at(x, y, 0.8f);
  c.appearance = colour;
  return c;
}
} // namespace

TEST_CASE("persistent identity re-identifies an object after a long gap", "[system][objectfile][identity]")
{
  // A red object is seen for three frames, then goes unobserved for 37 (past
  // max_inactive_age), and reappears after a bounce — nowhere near where
  // straight-line extrapolation puts it. A blue object stays in view.
  const cv::Vec3f red(60, 60, 230), blue(240, 110, 70);
  auto relabelled = [&](bool persistent)
  {
    system::ObjectFileStore::Config cfg;
    cfg.motion_prediction = true;
    cfg.appearance_matching = true;
    cfg.persistent_identity = persistent;
    system::ObjectFileStore store(cfg);
    for (int f = 0; f < 3; ++f)
    {
      store.update({coloured_at(100 + 6 * f, 100, red), coloured_at(400, 300, blue)}, f);
    }
    const int red_label = label_near(store, {112, 100});
    const int blue_label = label_near(store, {400, 300});
    for (int f = 3; f < 40; ++f)
    {
      store.update({coloured_at(400, 300, blue)}, f);
    }
    store.update({coloured_at(150, 180, red), coloured_at(400, 300, blue)}, 40);
    CHECK(label_near(store, {400, 300}) == blue_label);
    return label_near(store, {150, 180}) != red_label;
  };

  CHECK_FALSE(relabelled(true)); // same object file: identity held across the gap
  CHECK(relabelled(false));      // thesis default: aged out, a new file
}

TEST_CASE("persistent identity gives a newcomer of another colour its own file", "[system][objectfile][identity]")
{
  // Red vanishes; a green object appears where red was. Radius-only revival
  // hands the newcomer red's identity; the colour veto doesn't.
  const cv::Vec3f red(60, 60, 230), green(60, 200, 60);
  auto newcomer_label_is_red = [&](bool persistent)
  {
    system::ObjectFileStore::Config cfg;
    cfg.appearance_matching = true;
    cfg.persistent_identity = persistent;
    system::ObjectFileStore store(cfg);
    for (int f = 0; f < 3; ++f)
    {
      store.update({coloured_at(100, 100, red)}, f);
    }
    const int red_label = label_near(store, {100, 100});
    store.update({}, 3);
    store.update({coloured_at(104, 100, green)}, 4);
    return label_near(store, {104, 100}) == red_label;
  };

  CHECK_FALSE(newcomer_label_is_red(true));
  CHECK(newcomer_label_is_red(false));
}

TEST_CASE("attention_system config section: parsed, defaults kept, typos rejected", "[system][config]")
{
  system::AttentionSystem::Config cfg;
  system::AttentionSystem::apply_config_yaml(
      "segment_fraction: 0.25\n"
      "object_files:\n"
      "  persistent_identity: true\n"
      "  correspondence_radius: 45\n",
      cfg);
  CHECK(cfg.segment_fraction == 0.25f);
  CHECK(cfg.object_store.persistent_identity);
  CHECK(cfg.object_store.correspondence_radius == 45.0);
  CHECK(cfg.object_store.max_inactive_age == system::ObjectFileStore::Config{}.max_inactive_age);

  system::AttentionSystem::Config untouched;
  system::AttentionSystem::apply_config_yaml("", untouched);
  CHECK_FALSE(untouched.object_store.persistent_identity);

  CHECK_THROWS_AS(system::AttentionSystem::apply_config_yaml("object_files:\n  reid_color_gate: 20\n", cfg),
                  std::runtime_error);
  CHECK_THROWS_AS(system::AttentionSystem::apply_config_yaml("segmnt_fraction: 0.2\n", cfg), std::runtime_error);
}

TEST_CASE("segmentation: closing bridges an object's fragments, oversized regions are dropped", "[system][segment]")
{
  // Two crescents of one moving disk, 6 px apart, and a diffuse region
  // covering 60% of the map.
  cv::Mat saliency = cv::Mat::zeros(200, 300, CV_32F);
  cv::rectangle(saliency, cv::Rect(40, 40, 12, 30), cv::Scalar(1.0f), cv::FILLED);
  cv::rectangle(saliency, cv::Rect(58, 40, 12, 30), cv::Scalar(1.0f), cv::FILLED);
  cv::rectangle(saliency, cv::Rect(120, 0, 180, 200), cv::Scalar(0.6f), cv::FILLED);
  auto clusters = [&](int close, float max_fraction)
  {
    system::AttentionSystem::Config cfg;
    cfg.segment_close = close;
    cfg.max_cluster_fraction = max_fraction;
    return system::AttentionSystem(cfg).segment(saliency).size();
  };

  CHECK(clusters(0, 0.0f) == 3);  // thesis default: two fragments + the diffuse region
  CHECK(clusters(4, 0.0f) == 2);  // the fragments bridged into one object
  CHECK(clusters(4, 0.25f) == 1); // and the diffuse region dropped
}

TEST_CASE("AttentionSystem produces a scanpath over the motion sequence", "[system]")
{
  const fs::path dir = fs::path(ATTENTION_SOURCE_DIR) / "data" / "test_images" / "motion_seq";
  std::vector<std::string> frames = {(dir / "f00.png").string(), (dir / "f01.png").string(),
                                     (dir / "f02.png").string()};
  pipeline::ImageListSource source(frames);

  system::AttentionSystem::Config cfg; // defaults: exploration, scanpath mode
  system::AttentionSystem sys(cfg);

  int frames_seen = 0;
  sys.process_stream(source, [&](system::AttentionSystem&) { ++frames_seen; });

  CHECK(frames_seen == 3);
  CHECK_FALSE(sys.scanpath().empty());
  // Object files were formed from the salient square.
  CHECK_FALSE(sys.active_files().empty());
}

TEST_CASE("IOR-ablation behaviors differ by inhibition domain", "[system][behavior][ior]")
{
  // The dynamic-IOR study (M12) hinges on these three differing only in what
  // they inhibit. Two objects; the "strong" one stays the more salient.
  auto make_store = [] { return system::ObjectFileStore{}; };

  SECTION("greedy (no IOR) perseverates on the most salient object")
  {
    auto behavior = system::create_behavior("greedy");
    system::ObjectFileStore store = make_store();
    const auto strong = cluster_at(20, 20, 0.9f);
    const auto weak = cluster_at(150, 150, 0.5f);
    int strong_label = 0;
    for (int f = 0; f < 5; ++f)
    {
      store.update({strong, weak}, f);
      if (f == 0)
      {
        strong_label = label_near(store, {20, 20});
      }
      const system::ObjectFile* focus = behavior->select_focus(store, f);
      REQUIRE(focus != nullptr);
      CHECK(focus->label == strong_label); // never leaves the strongest
    }
  }

  SECTION("object-ior leaves a static object after selecting it")
  {
    auto behavior = system::create_behavior("object-ior");
    system::ObjectFileStore store = make_store();
    const auto strong = cluster_at(20, 20, 0.9f);
    const auto weak = cluster_at(150, 150, 0.5f);

    store.update({strong, weak}, 0);
    const int strong_label = label_near(store, {20, 20});
    const system::ObjectFile* f0 = behavior->select_focus(store, 0);
    REQUIRE(f0 != nullptr);
    CHECK(f0->label == strong_label); // strongest first

    store.update({strong, weak}, 1);
    const system::ObjectFile* f1 = behavior->select_focus(store, 1);
    REQUIRE(f1 != nullptr);
    CHECK(f1->label != strong_label); // inhibited -> moves to the other object
  }

  SECTION("spatial-ior inhibits a selected location (unlike greedy)")
  {
    // On static objects space-based IOR behaves like object-based: it leaves the
    // strong object after looking at it. (The two diverge only under motion,
    // which is an emergent, statistical effect measured by the M12 study, not a
    // single-step unit assertion.)
    auto behavior = system::create_behavior("spatial-ior");
    system::ObjectFileStore store = make_store();
    const auto strong = cluster_at(20, 20, 0.9f);
    const auto weak = cluster_at(150, 150, 0.5f);

    store.update({strong, weak}, 0);
    const int strong_label = label_near(store, {20, 20});
    REQUIRE(behavior->select_focus(store, 0)->label == strong_label);

    store.update({strong, weak}, 1);
    CHECK(behavior->select_focus(store, 1)->label != strong_label); // location inhibited
  }
}

TEST_CASE("AttentionSystem in Feature mode keeps no object files", "[system]")
{
  const fs::path dir = fs::path(ATTENTION_SOURCE_DIR) / "data" / "test_images" / "motion_seq";
  std::vector<std::string> frames = {(dir / "f00.png").string(), (dir / "f01.png").string()};
  pipeline::ImageListSource source(frames);

  system::AttentionSystem::Config cfg;
  cfg.action_mode = system::AttentionSystem::ActionMode::Feature;
  system::AttentionSystem sys(cfg);
  sys.process_stream(source);

  CHECK(sys.scanpath().empty());
  CHECK(sys.current_focus() == nullptr);
}
