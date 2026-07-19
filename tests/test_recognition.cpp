// M13 tests: recognition on attended regions — label memory on object files,
// the Identification behavior (semantic IOR / curiosity), the parameterized
// processor registry, and the timed processor runner.

#include "attention/system/attention_system.h"
#include "attention/system/behavior.h"
#include "attention/system/object_file.h"
#include "attention/system/processor.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

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

// A deterministic processor for runner tests: one fixed ROI-relative hit.
class FixedHit : public system::Processor
{
 public:
  std::string name() const override { return "fixed-hit"; }

  system::Annotation process(const system::ObjectFile& object, const cv::Mat& roi) const override
  {
    system::Annotation a;
    a.processor = name();
    a.object_label = object.label;
    a.class_label = "widget";
    a.confidence = 0.9f;
    a.detections.push_back({cv::Rect(2, 3, 4, 5), 0.9f, "widget"});
    (void)roi;
    return a;
  }
};
} // namespace

TEST_CASE("label memory: majority vote, confidence, tie-break", "[recognition][labels]")
{
  system::LabelMemory memory;
  CHECK(memory.best_label().empty());
  CHECK(memory.best_confidence() == 0.0f);

  memory.add_vote("person", 0.8f);
  memory.add_vote("person", 0.6f);
  memory.add_vote("car", 0.9f);
  CHECK(memory.best_label() == "person");
  CHECK(memory.best_count() == 2);
  CHECK(memory.best_confidence() == Catch::Approx(0.7f).margin(1e-5));

  system::LabelMemory tie;
  tie.add_vote("a", 0.5f);
  tie.add_vote("b", 0.9f);
  CHECK(tie.best_label() == "b"); // equal counts: higher confidence sum wins
}

TEST_CASE("object-file store accumulates votes and inspections", "[recognition][labels]")
{
  system::ObjectFileStore store;
  store.update({cluster_at(50, 50, 0.5f)}, 0);
  const int label = store.active_files()[0].label;

  store.record_inspection(label);
  store.add_label_vote(label, "person", 0.8f);
  store.add_label_vote(label, "", 0.9f); // empty class = no vote

  const system::ObjectFile* file = store.find_active(label);
  REQUIRE(file != nullptr);
  CHECK(file->labels.inspections == 1);
  CHECK(file->labels.best_label() == "person");
  CHECK(file->labels.best_count() == 1);

  // Label memory survives the inactive -> revived cycle (persistent identity).
  store.update({}, 1);                         // vanishes -> inactive
  store.update({cluster_at(52, 50, 0.5f)}, 2); // reappears nearby -> revived
  const system::ObjectFile* revived = store.find_active(label);
  REQUIRE(revived != nullptr);
  CHECK(revived->labels.best_label() == "person");
}

TEST_CASE("Identification prefers the unknown, drops the identified", "[recognition][behavior]")
{
  system::ObjectFileStore store;
  store.update({cluster_at(20, 20, 0.9f), cluster_at(100, 100, 0.5f)}, 0);
  const int first = store.active_files()[0].label;
  const int second = store.active_files()[1].label;

  system::Identification::Params params;
  params.min_votes = 1;
  params.confidence = 0.5f;
  params.max_inspections = 2;
  params.dwell_frames = 1;
  system::Identification behavior(params);

  // Both unknown: the more salient one leads.
  const system::ObjectFile* focus = behavior.select_focus(store, 0);
  REQUIRE(focus != nullptr);
  CHECK(focus->label == first);

  // Confidently labeling it makes it settled — curiosity moves on.
  store.record_inspection(first);
  store.add_label_vote(first, "person", 0.9f);
  CHECK(behavior.is_settled(*store.find_active(first)));
  focus = behavior.select_focus(store, 1);
  REQUIRE(focus != nullptr);
  CHECK(focus->label == second);

  // Fruitless inspections eventually settle an object too (give-up, no
  // perseveration on unlabelable objects).
  store.record_inspection(second);
  store.record_inspection(second);
  CHECK(behavior.is_settled(*store.find_active(second)));

  // Everything settled: the behavior still produces a focus (background
  // rotation, exploration-style) rather than starving.
  CHECK(behavior.select_focus(store, 2) != nullptr);
}

TEST_CASE("processor registry parses name:config specs", "[recognition][processor]")
{
  auto probe = system::create_processor("roi-probe");
  REQUIRE(probe != nullptr);
  CHECK(probe->name() == "roi-probe");

  // Unknown names throw with the available list.
  CHECK_THROWS_AS(system::create_processor("no-such-processor"), std::runtime_error);

  // A config string reaches the factory: dnn-classify fails fast on a missing
  // model file (with the fetch hint), proving the spec was parsed and used.
  CHECK_THROWS_WITH(system::create_processor("dnn-classify:/nonexistent/model.onnx:labels.txt"),
                    Catch::Matchers::ContainsSubstring("fetch_models"));
}

TEST_CASE("run_processor fills accounting and maps detections to image coordinates", "[recognition][processor]")
{
  system::ObjectFile object;
  object.label = 7;
  cv::Mat roi(40, 30, CV_8UC3, cv::Scalar(10, 20, 30));

  FixedHit processor;
  const system::Annotation a = system::run_processor(processor, object, roi, cv::Point(100, 200), 5);

  CHECK(a.frame == 5);
  CHECK(a.pixels == 40 * 30);
  CHECK(a.ms >= 0.0);
  REQUIRE(a.detections.size() == 1);
  CHECK(a.detections[0].box == cv::Rect(102, 203, 4, 5)); // ROI-relative + origin
  CHECK(a.class_label == "widget");
}

TEST_CASE("hog-person answers gracefully on arbitrary ROIs", "[recognition][processor]")
{
  auto hog = system::create_processor("hog-person");
  system::ObjectFile object;
  object.label = 3;

  // Empty and tiny ROIs must not crash and must still annotate the object.
  const system::Annotation empty = hog->process(object, cv::Mat());
  CHECK(empty.object_label == 3);
  CHECK(empty.class_label.empty());

  cv::Mat noise(60, 40, CV_8UC3);
  cv::randu(noise, 0, 255);
  const system::Annotation on_noise = hog->process(object, noise);
  CHECK(on_noise.processor == "hog-person");
  CHECK_FALSE(on_noise.label.empty());
}

TEST_CASE("gated cadence: per-dwell fires once per dwell window", "[recognition][system]")
{
  // Drive the AttentionSystem directly with synthetic frames: one bright block
  // on black yields one object; the focus stays on it, so PerDwell must fire
  // once per process_repeat_frames window (a held focus is a *sequence* of
  // attentive computations — otherwise a persistent object could never gather
  // enough label votes to settle) while EveryFrame fires each frame.
  auto make_frame = []()
  {
    cv::Mat image(120, 160, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(image, cv::Rect(60, 40, 30, 30), cv::Scalar(255, 255, 255), cv::FILLED);
    return image;
  };

  for (const bool every_frame : {false, true})
  {
    system::AttentionSystem::Config config;
    pipeline::FeatureSpec intensity;
    intensity.type = "intensity";
    config.pipeline.features = {intensity};
    config.processors = {"roi-probe"};
    config.processor_cadence = every_frame ? system::AttentionSystem::ProcessorCadence::EveryFrame
                                           : system::AttentionSystem::ProcessorCadence::PerDwell;
    system::AttentionSystem sys(config);
    sys.reset();
    for (int i = 0; i < 7; ++i)
    {
      sys.process_frame(make_frame());
    }
    const long long calls = sys.processor_stats().count("roi-probe") ? sys.processor_stats().at("roi-probe").calls : 0;
    if (every_frame)
    {
      CHECK(calls >= 6); // once per focused frame
    }
    else
    {
      // 7 frames held on one object with process_repeat_frames = 3: the dwell
      // re-fires at the start of each window (frames 0, 3, 6) — a held focus
      // is a sequence of attentive computations, not a single one (otherwise a
      // persistent object could never gather enough label votes to settle).
      CHECK(calls == 3);
    }
  }
}
