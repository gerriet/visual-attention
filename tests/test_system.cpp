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

TEST_CASE("proto-objects: one cluster per object, touching objects apart, background dropped",
          "[system][segment][proto]")
{
  // Two touching disks (red, blue) on a dark ground, salient only along two
  // broken onset arcs each, plus a salient blob over empty background.
  cv::Mat image(200, 300, CV_8UC3, cv::Scalar(25, 25, 25));
  const cv::Point red_centre(100, 100), blue_centre(139, 100);
  const cv::Scalar red(60, 60, 230), blue(240, 110, 70); // BGR
  cv::circle(image, red_centre, 20, red, cv::FILLED);
  cv::circle(image, blue_centre, 20, blue, cv::FILLED);
  cv::Mat saliency = cv::Mat::zeros(image.size(), CV_32F);
  for (const cv::Point& centre : {red_centre, blue_centre})
  {
    cv::ellipse(saliency, centre, cv::Size(20, 20), 0, 10, 160, cv::Scalar(1.0), 3);
    cv::ellipse(saliency, centre, cv::Size(20, 20), 0, 190, 340, cv::Scalar(1.0), 3);
  }
  cv::circle(saliency, cv::Point(250, 50), 10, cv::Scalar(0.8), cv::FILLED);

  system::AttentionSystem::Config cfg;
  // Thesis path: the touching disks' arcs meet at the contact point, so it
  // yields two clusters that each span *both* objects, plus the blob.
  CHECK(system::AttentionSystem(cfg).segment(saliency, image).size() == 3);

  cfg.proto_objects = true;
  const auto objects = system::AttentionSystem(cfg).segment(saliency, image);
  REQUIRE(objects.size() == 2);
  for (const auto& object : objects)
  {
    const bool is_red = cv::norm(object.centroid - red_centre) < cv::norm(object.centroid - blue_centre);
    CHECK(cv::norm(object.centroid - (is_red ? red_centre : blue_centre)) <= 2.0);
    const cv::Scalar colour = is_red ? red : blue;
    CHECK(cv::norm(object.appearance - cv::Vec3f(static_cast<float>(colour[0]), static_cast<float>(colour[1]),
                                                 static_cast<float>(colour[2]))) < 10.0);
  }
}

TEST_CASE("persistent identity folds a second file on one object into the first", "[system][objectfile][identity]")
{
  // One red object; on frame 1 a second, look-alike cluster appears on it (a
  // partial fragment). Without merging, both files persist.
  const cv::Vec3f red(60, 60, 230);
  auto active_after = [&](bool persistent)
  {
    system::ObjectFileStore::Config cfg;
    cfg.appearance_matching = true;
    cfg.persistent_identity = persistent;
    system::ObjectFileStore store(cfg);
    system::Cluster whole = coloured_at(100, 100, red);
    whole.bbox = cv::Rect(90, 90, 20, 20);
    store.update({whole}, 0);
    const int label = store.active_files().front().label;
    system::Cluster fragment = coloured_at(104, 102, red);
    fragment.bbox = cv::Rect(100, 98, 8, 8);
    store.update({whole, fragment}, 1);
    return std::make_pair(store.active_files().size(), label == store.active_files().front().label);
  };

  const auto persistent = active_after(true);
  CHECK(persistent.first == 1u);
  CHECK(persistent.second); // the older label survives
  CHECK(active_after(false).first == 2u);
}

TEST_CASE("persistent identity folds an inactive duplicate into the active file", "[system][objectfile][identity]")
{
  // A second, look-alike file is born on a fragment just beside the object
  // (no box overlap), then the fragment vanishes: its inactive file would
  // otherwise wait to be revived on the object instead of the real file.
  const cv::Vec3f red(60, 60, 230);
  auto inactive_after = [&](bool persistent)
  {
    system::ObjectFileStore::Config cfg;
    cfg.appearance_matching = true;
    cfg.persistent_identity = persistent;
    system::ObjectFileStore store(cfg);
    system::Cluster whole = coloured_at(100, 100, red);
    whole.bbox = cv::Rect(90, 90, 20, 20);
    system::Cluster fragment = coloured_at(100, 125, red);
    fragment.bbox = cv::Rect(95, 120, 10, 10);
    store.update({whole}, 0);
    store.update({whole, fragment}, 1);
    store.update({whole}, 2);
    return store.inactive_files().size();
  };

  CHECK(inactive_after(true) == 0u);
  CHECK(inactive_after(false) == 1u);
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

TEST_CASE("motion-compensated spatial IOR: the tag travels with the object it was left on", "[system][behavior][ior]")
{
  // The strengthened space-based baseline of the H1 study. A strong object
  // moves 25 px per frame past a tight (15 px) tag; a weak one stands still.
  // A plain location tag is left behind, so the strong object wins again and
  // again; a tag that keeps the object's velocity stays on it — without using
  // the object's identity after the deposit.
  system::IorBehavior::Params params;
  params.ior_radius = 15.0f;

  auto focus_at_frame_2 = [&](const std::string& name)
  {
    auto behavior = system::create_behavior(name, params);
    system::ObjectFileStore store;
    int strong_label = -1;
    const system::ObjectFile* focus = nullptr;
    for (int f = 0; f < 3; ++f)
    {
      store.update({cluster_at(20 + 25 * f, 40, 0.9f), cluster_at(150, 150, 0.5f)}, f);
      if (f == 0)
      {
        strong_label = label_near(store, {20, 40});
      }
      focus = behavior->select_focus(store, f);
      REQUIRE(focus != nullptr);
    }
    REQUIRE(label_near(store, {70, 40}) == strong_label); // identity held; only the tag differs
    return focus->label == strong_label;
  };

  CHECK(focus_at_frame_2("spatial-ior"));          // escaped its tag: re-fixated
  CHECK_FALSE(focus_at_frame_2("spatial-ior-mc")); // the tag kept up: attention moves on
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

// --- replication-v2: the correspondence of thesis §7.2.3, and object files on the
// --- neural field's activity clusters (docs/replication/WAPCV_2003_NOTES.md)

namespace
{
system::Cluster featured(int x, int y, std::vector<float> features)
{
  system::Cluster c = cluster_at(x, y, 0.5f);
  c.features = std::move(features);
  return c;
}

system::ObjectFileStore thesis_store(double radius = 8.0, int max_age = 30)
{
  system::ObjectFileStore::Config config;
  config.rule = system::ObjectFileStore::Rule::Thesis;
  config.correspondence_radius = radius;
  config.max_inactive_age = max_age;
  return system::ObjectFileStore(config);
}
} // namespace

TEST_CASE("thesis correspondence: position decides where there are no features", "[system][objectfile][thesis]")
{
  system::ObjectFileStore store = thesis_store();
  store.update({cluster_at(20, 20, 0.5f), cluster_at(60, 20, 0.5f)}, 0);
  const int left = label_near(store, {20, 20});
  const int right = label_near(store, {60, 20});
  REQUIRE(left != 0);
  REQUIRE(right != 0);

  // Both move a little: unambiguous within the radius
  store.update({cluster_at(23, 21, 0.5f), cluster_at(57, 20, 0.5f)}, 1);
  CHECK(label_near(store, {23, 21}) == left);
  CHECK(label_near(store, {57, 20}) == right);

  // A jump beyond the radius but within twice of it: still the nearest file
  store.update({cluster_at(35, 21, 0.5f), cluster_at(57, 20, 0.5f)}, 2);
  CHECK(label_near(store, {35, 21}) == left);

  // Beyond twice the radius it is a new object; the old file goes inactive
  store.update({cluster_at(35, 60, 0.5f), cluster_at(57, 20, 0.5f)}, 3);
  CHECK(label_near(store, {35, 60}) != left);
  CHECK(store.inactive_files().size() == 1);
}

TEST_CASE("thesis correspondence: a cluster gets the file that is nearest and most similar",
          "[system][objectfile][thesis]")
{
  // Two objects pass each other inside the radius: every cluster is within
  // reach of both files, so position alone is ambiguous
  const std::vector<float> red = {0.9f, 0.1f}, blue = {0.1f, 0.9f};

  SECTION("nearest and most similar agree: the files carry on")
  {
    system::ObjectFileStore store = thesis_store(12.0);
    store.update({featured(20, 20, red), featured(30, 20, blue)}, 0);
    const int a = label_near(store, {20, 20}, 2.0);
    const int b = label_near(store, {30, 20}, 2.0);
    store.update({featured(23, 20, red), featured(27, 20, blue)}, 1);
    CHECK(label_near(store, {23, 20}, 1.0) == a);
    CHECK(label_near(store, {27, 20}, 1.0) == b);
    CHECK(store.inactive_files().empty());
  }
  SECTION("they disagree (the objects have swapped places): the features win, via the inactive files")
  {
    system::ObjectFileStore store = thesis_store(12.0);
    store.update({featured(20, 20, red), featured(30, 20, blue)}, 0);
    const int a = label_near(store, {20, 20}, 2.0);
    const int b = label_near(store, {30, 20}, 2.0);
    store.update({featured(22, 20, blue), featured(28, 20, red)}, 1);
    CHECK(label_near(store, {22, 20}, 1.0) == b);
    CHECK(label_near(store, {28, 20}, 1.0) == a);
  }
}

TEST_CASE("thesis correspondence: inactive files are revived primarily by their features",
          "[system][objectfile][thesis]")
{
  system::ObjectFileStore store = thesis_store();
  store.update({featured(20, 20, {0.9f, 0.1f})}, 0);
  const int original = label_near(store, {20, 20});
  store.update({}, 1); // gone
  REQUIRE(store.inactive_files().size() == 1);

  SECTION("the same features, somewhere else: the same object")
  {
    store.update({featured(120, 90, {0.88f, 0.12f})}, 5);
    CHECK(label_near(store, {120, 90}) == original);
    CHECK(store.inactive_files().empty());
  }
  SECTION("other features at the old place: a new object")
  {
    store.update({featured(20, 20, {0.1f, 0.9f})}, 5);
    CHECK(label_near(store, {20, 20}) != original);
    CHECK(store.inactive_files().size() == 1);
  }
  SECTION("beyond the maximum age the file is gone")
  {
    store.update({}, 40);
    CHECK(store.inactive_files().empty());
    store.update({featured(20, 20, {0.9f, 0.1f})}, 41);
    CHECK(label_near(store, {20, 20}) != original);
  }
}

TEST_CASE("thesis correspondence: merged clusters are resolved by features after four frames",
          "[system][objectfile][thesis]")
{
  system::ObjectFileStore store = thesis_store();
  store.update({featured(20, 20, {0.9f, 0.1f}), featured(34, 20, {0.1f, 0.9f})}, 0);
  const int a = label_near(store, {20, 20});
  const int b = label_near(store, {34, 20});

  // The two clusters merge into one that looks like the first
  for (int frame = 1; frame <= 4; ++frame)
  {
    store.update({featured(27, 20, {0.85f, 0.15f})}, frame);
    REQUIRE(store.active_files().size() == 1);
    if (frame < 4)
    {
      CHECK(store.active_files()[0].label != a);
      CHECK(store.active_files()[0].merged_from.size() == 2);
    }
  }
  store.update({featured(27, 20, {0.85f, 0.15f})}, 5);
  CHECK(store.active_files()[0].label == a); // it was the first all along
  CHECK(store.active_files()[0].merged_from.empty());
  // The other predecessor stays available for revival
  bool b_inactive = false;
  for (const auto& file : store.inactive_files())
  {
    b_inactive = b_inactive || file.label == b;
  }
  CHECK(b_inactive);
}

TEST_CASE("the position rule is unchanged by the new fields", "[system][objectfile]")
{
  system::ObjectFileStore store; // default: Rule::Position
  store.update({featured(20, 20, {0.9f, 0.1f})}, 0);
  const int label = label_near(store, {20, 20});
  store.update({featured(24, 20, {0.1f, 0.9f})}, 1); // features are not consulted
  CHECK(label_near(store, {24, 20}) == label);
}

TEST_CASE("attention_system config: cluster source and correspondence rule", "[system][config][thesis]")
{
  system::AttentionSystem::Config cfg;
  CHECK(cfg.cluster_source == system::AttentionSystem::Config::ClusterSource::Saliency);
  CHECK(cfg.object_store.rule == system::ObjectFileStore::Rule::Position);
  system::AttentionSystem::apply_config_yaml(
      "cluster_source: field\nobject_files:\n  correspondence: thesis\n  feature_gate: 0.2\n", cfg);
  CHECK(cfg.cluster_source == system::AttentionSystem::Config::ClusterSource::Field);
  CHECK(cfg.object_store.rule == system::ObjectFileStore::Rule::Thesis);
  CHECK(cfg.object_store.feature_gate == 0.2);
  CHECK_THROWS_AS(system::AttentionSystem::apply_config_yaml("cluster_source: fields\n", cfg), std::runtime_error);
  CHECK_THROWS_AS(system::AttentionSystem::apply_config_yaml("object_files:\n  correspondence: nearest\n", cfg),
                  std::runtime_error);
}

TEST_CASE("camera compensation: stored coordinates follow a panning camera", "[system][camera]")
{
  // A textured scene shifted by a known translation from frame to frame: the
  // estimated shift must equal it in sign and size, and object files must move
  // with the scene.
  cv::Mat scene(400, 500, CV_8UC3);
  cv::RNG rng(7);
  rng.fill(scene, cv::RNG::UNIFORM, 0, 255);
  cv::GaussianBlur(scene, scene, cv::Size(0, 0), 2.0);
  cv::circle(scene, cv::Point(250, 200), 25, cv::Scalar(0, 0, 255), -1);

  system::AttentionSystem::Config cfg;
  system::AttentionSystem::apply_config_yaml("camera_compensation: true\n", cfg);
  cfg.pipeline.selection = "nms";
  system::AttentionSystem sys(cfg);
  sys.reset();
  const cv::Point2f pan(6.0f, -4.0f); // the camera pans: the scene moves by this per frame
  int disk_label = 0;
  cv::Point disk_at_start;
  for (int t = 0; t < 4; ++t)
  {
    const cv::Point2f offset = pan * static_cast<float>(t);
    const cv::Mat warp = (cv::Mat_<double>(2, 3) << 1, 0, offset.x, 0, 1, offset.y);
    cv::Mat frame;
    cv::warpAffine(scene, frame, warp, scene.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    frame = frame(cv::Rect(40, 40, 400, 300)).clone(); // crop away the reflected borders
    sys.process_frame(frame);
    if (t == 0)
    {
      // The file nearest the disk (at 250 - 40, 200 - 40 in the crop)
      double best = 1e9;
      for (const auto& file : sys.active_files())
      {
        const double d = cv::norm(file.centroid - cv::Point(210, 160));
        if (d < best)
        {
          best = d;
          disk_label = file.label;
          disk_at_start = file.centroid;
        }
      }
      REQUIRE(best < 40.0);
    }
    else
    {
      CHECK(std::abs(sys.last_camera_shift().x - pan.x) < 1.0f);
      CHECK(std::abs(sys.last_camera_shift().y - pan.y) < 1.0f);
    }
  }
  // The same file, three pans later, has moved with the scene
  bool kept = false;
  for (const auto& file : sys.active_files())
  {
    if (file.label == disk_label)
    {
      kept = true;
      const cv::Point expected = disk_at_start + cv::Point(18, -12);
      // (the centroid is the current cluster's, so segmentation jitter adds to this)
      CHECK(cv::norm(file.centroid - expected) < 20.0);
    }
  }
  CHECK(kept);
}
