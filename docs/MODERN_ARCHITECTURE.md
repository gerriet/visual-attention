# Modern Visual Attention System - Architecture Design

## Project Overview

A modern, extensible visual attention framework that:
- Processes single images, video sequences, and stereo pairs
- Supports pluggable feature extraction, integration, and tracking mechanisms
- Provides multiple visualization options
- Integrates with 3rd-party object recognition tools
- Offers both C++ and Python APIs

---

## 1. **Technology Stack**

### Core
- **C++17/20**: Modern language features (smart pointers, move semantics, concepts)
- **CMake 3.20+**: Cross-platform build system
- **Conan/vcpkg**: Package management

### Key Libraries

**Computer Vision:**
- **OpenCV 4.x**: Image I/O, basic processing, camera access
- **Eigen3**: Linear algebra, matrix operations
- **PCL** (optional): Point cloud processing for depth data

**Framework:**
- **Boost 1.75+**:
  - `boost::signals2` for event-driven visualization
  - `boost::property_tree` for configuration
  - `boost::dll` for plugin system
- **pybind11**: Python bindings
- **spdlog**: Modern logging
- **nlohmann/json** or **yaml-cpp**: Configuration files

**Optional/Advanced:**
- **TensorFlow/PyTorch C++**: For DNN-based features
- **ROS2** (optional): If robotic integration needed
- **OpenMP/TBB**: Parallelization

---

## 2. **Architecture Overview**

### **Design Principles**

1. **Plugin Architecture**: Features, integrators, trackers as plugins
2. **Pipeline Pattern**: Data flows through configurable processing stages
3. **Strategy Pattern**: Swappable algorithms
4. **Observer Pattern**: Visualization and logging
5. **Dependency Injection**: Runtime configuration
6. **Interface Segregation**: Small, focused interfaces

### **High-Level Structure**

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ C++ API      │  │ Python API   │  │ CLI/GUI Tools    │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Core Framework Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Pipeline     │  │ Plugin       │  │ Configuration    │  │
│  │ Engine       │  │ Manager      │  │ System           │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Processing Components Layer                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Input        │→ │ Feature      │→ │ Integration      │  │
│  │ Sources      │  │ Extraction   │  │ & Saliency       │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                              ↓               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Visualization│← │ Object       │← │ Attention        │  │
│  │ & Output     │  │ Recognition  │  │ Selection        │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. **Core Module Design**

### **3.1 Data Model**

```cpp
// Modern data structures using smart pointers and move semantics

namespace attention {

// Frame represents a single time point with possible stereo
struct Frame {
    cv::Mat left;                          // Primary image
    std::optional<cv::Mat> right;          // Optional stereo pair
    std::optional<cv::Mat> depth;          // Optional depth map
    int64_t timestamp_us;                  // Microseconds
    std::unordered_map<std::string, cv::Mat> metadata; // e.g., optical flow

    bool is_stereo() const { return right.has_value(); }
    bool has_depth() const { return depth.has_value(); }
};

// FeatureMap represents a computed feature
struct FeatureMap {
    std::string name;                      // e.g., "color", "symmetry"
    cv::Mat data;                          // Feature values (float)
    cv::Rect roi;                          // Region of interest
    double confidence = 1.0;               // Overall confidence

    // Metadata for visualization/debugging
    std::unordered_map<std::string, std::any> properties;
};

// SaliencyMap represents integrated attention
struct SaliencyMap {
    cv::Mat map;                           // Float saliency values [0,1]
    std::vector<cv::Point> peaks;          // Local maxima
    std::optional<cv::Point> focus;        // Current focus of attention

    // History for temporal coherence
    std::deque<cv::Mat> history;
};

// AttentionState holds complete system state
struct AttentionState {
    Frame current_frame;
    std::vector<FeatureMap> features;
    SaliencyMap saliency;
    std::vector<TrackedObject> objects;

    // Temporal context
    std::optional<AttentionState> previous;
};

} // namespace attention
```

### **3.2 Plugin Interface System**

```cpp
// Base interface for all plugins
class IPlugin {
public:
    virtual ~IPlugin() = default;
    virtual std::string name() const = 0;
    virtual std::string version() const = 0;
    virtual void configure(const Config& config) = 0;
};

// Feature extraction interface
class IFeatureExtractor : public IPlugin {
public:
    enum class InputType { MONO, STEREO, SEQUENCE, DEPTH };

    virtual ~IFeatureExtractor() = default;

    // What kind of input does this feature need?
    virtual InputType input_type() const = 0;

    // Synchronous computation
    virtual FeatureMap compute(const Frame& frame) = 0;

    // Async computation (returns future)
    virtual std::future<FeatureMap> compute_async(const Frame& frame) {
        return std::async(std::launch::async,
                         [this, &frame]() { return compute(frame); });
    }

    // For sequence-based features (motion, tracking)
    virtual FeatureMap compute_sequence(const std::vector<Frame>& frames) {
        throw std::runtime_error("Sequence processing not supported");
    }
};

// Feature integration interface (combines multiple features)
class IFeatureIntegrator : public IPlugin {
public:
    virtual ~IFeatureIntegrator() = default;

    virtual SaliencyMap integrate(
        const std::vector<FeatureMap>& features,
        const std::optional<SaliencyMap>& prior = std::nullopt
    ) = 0;

    // Set feature weights dynamically
    virtual void set_weights(const std::unordered_map<std::string, double>& weights) = 0;
};

// Attention selection interface (WTA, IOR, neural fields, etc.)
class IAttentionSelector : public IPlugin {
public:
    virtual ~IAttentionSelector() = default;

    // Update internal state and select next focus
    virtual cv::Point select(
        const SaliencyMap& saliency,
        const AttentionState& state
    ) = 0;

    // Reset inhibition of return, memory, etc.
    virtual void reset() = 0;
};

// Object recognition interface (3rd party integration)
class IObjectRecognizer : public IPlugin {
public:
    struct Detection {
        cv::Rect bbox;
        std::string label;
        double confidence;
        std::unordered_map<std::string, std::any> features;
    };

    virtual ~IObjectRecognizer() = default;

    virtual std::vector<Detection> recognize(
        const Frame& frame,
        const cv::Rect& roi
    ) = 0;
};

// Visualization interface
class IVisualizer : public IPlugin {
public:
    enum class Mode { STATIC, ANIMATED, HEATMAP, SCANPATH };

    virtual ~IVisualizer() = default;

    virtual void visualize(
        const AttentionState& state,
        const Mode mode = Mode::STATIC
    ) = 0;

    virtual cv::Mat render() = 0;

    // For saving animations, generating reports
    virtual void export_result(const std::string& path) = 0;
};
```

### **3.3 Plugin Manager**

```cpp
class PluginManager {
public:
    // Load plugins from directory
    void load_plugins(const std::filesystem::path& plugin_dir);

    // Load single plugin
    void load_plugin(const std::filesystem::path& plugin_path);

    // Get plugin by name and type
    template<typename T>
    std::shared_ptr<T> get_plugin(const std::string& name) {
        static_assert(std::is_base_of_v<IPlugin, T>);

        auto it = plugins_.find(name);
        if (it == plugins_.end()) {
            throw std::runtime_error("Plugin not found: " + name);
        }

        auto plugin = std::dynamic_pointer_cast<T>(it->second);
        if (!plugin) {
            throw std::runtime_error("Invalid plugin type: " + name);
        }

        return plugin;
    }

    // List available plugins by type
    template<typename T>
    std::vector<std::string> list_plugins() const;

private:
    std::unordered_map<std::string, std::shared_ptr<IPlugin>> plugins_;
    std::vector<boost::dll::shared_library> loaded_libs_;
};
```

### **3.4 Processing Pipeline**

```cpp
class AttentionPipeline {
public:
    // Builder pattern for configuration
    class Builder {
    public:
        Builder& add_input_source(std::shared_ptr<IInputSource> source);
        Builder& add_feature(std::shared_ptr<IFeatureExtractor> feature);
        Builder& set_integrator(std::shared_ptr<IFeatureIntegrator> integrator);
        Builder& set_selector(std::shared_ptr<IAttentionSelector> selector);
        Builder& add_recognizer(std::shared_ptr<IObjectRecognizer> recognizer);
        Builder& add_visualizer(std::shared_ptr<IVisualizer> visualizer);
        Builder& set_config(const Config& config);

        std::unique_ptr<AttentionPipeline> build();

    private:
        // ... builder state
    };

    // Pipeline execution
    void process_frame();
    void process_sequence(size_t num_frames);
    void run_continuous(); // For video/camera

    // Access current state
    const AttentionState& state() const { return state_; }

    // Event system for visualization and monitoring
    boost::signals2::signal<void(const AttentionState&)> on_state_update;
    boost::signals2::signal<void(const FeatureMap&)> on_feature_computed;
    boost::signals2::signal<void(const cv::Point&)> on_attention_shift;

private:
    std::shared_ptr<IInputSource> input_;
    std::vector<std::shared_ptr<IFeatureExtractor>> features_;
    std::shared_ptr<IFeatureIntegrator> integrator_;
    std::shared_ptr<IAttentionSelector> selector_;
    std::vector<std::shared_ptr<IObjectRecognizer>> recognizers_;
    std::vector<std::shared_ptr<IVisualizer>> visualizers_;

    AttentionState state_;
    Config config_;

    // Thread pool for parallel feature extraction
    std::unique_ptr<ThreadPool> thread_pool_;
};
```

---

## 4. **Example Plugins Implementation**

### **4.1 Stereo Depth Feature**

```cpp
class StereoDepthFeature : public IFeatureExtractor {
public:
    std::string name() const override { return "stereo_depth"; }
    std::string version() const override { return "1.0.0"; }

    InputType input_type() const override { return InputType::STEREO; }

    void configure(const Config& config) override {
        min_disparity_ = config.get<int>("min_disparity", -25);
        max_disparity_ = config.get<int>("max_disparity", 0);
        block_size_ = config.get<int>("block_size", 15);

        // OpenCV stereo matcher
        matcher_ = cv::StereoSGBM::create(
            min_disparity_, max_disparity_ - min_disparity_, block_size_
        );
    }

    FeatureMap compute(const Frame& frame) override {
        if (!frame.is_stereo()) {
            throw std::runtime_error("Stereo input required");
        }

        cv::Mat disparity;
        matcher_->compute(frame.left, *frame.right, disparity);

        // Convert to normalized saliency
        cv::Mat saliency = compute_depth_saliency(disparity);

        FeatureMap result;
        result.name = name();
        result.data = saliency;
        result.properties["disparity"] = disparity;

        return result;
    }

private:
    cv::Ptr<cv::StereoSGBM> matcher_;
    int min_disparity_, max_disparity_, block_size_;

    cv::Mat compute_depth_saliency(const cv::Mat& disparity) {
        // Implement depth pop-out based on disparity gradients
        // Similar to original code but with modern OpenCV
        // ...
    }
};

// Plugin export macro
ATTENTION_PLUGIN_EXPORT(StereoDepthFeature)
```

### **4.2 Neural Field Integrator**

```cpp
class NeuralFieldIntegrator : public IFeatureIntegrator {
public:
    void configure(const Config& config) override {
        field_size_ = config.get<int>("field_size", 64);
        alpha_ = config.get<double>("alpha", 0.5);
        beta_ = config.get<double>("beta", 30.0);
        rest_value_ = config.get<double>("rest_value", -0.25);

        // Initialize 2D or 3D field based on config
        if (config.get<bool>("use_3d", false)) {
            field_3d_ = std::make_unique<NeuralField3D>(
                field_size_, field_size_,
                config.get<int>("depth_layers", 10)
            );
        } else {
            field_2d_ = std::make_unique<NeuralField2D>(
                field_size_, field_size_
            );
        }

        setup_kernels(config);
    }

    SaliencyMap integrate(
        const std::vector<FeatureMap>& features,
        const std::optional<SaliencyMap>& prior
    ) override {
        // Combine weighted features
        cv::Mat combined = combine_features(features);

        // Update neural field dynamics
        if (field_3d_) {
            field_3d_->update(combined, num_iterations_);
            return field_3d_->get_saliency();
        } else {
            field_2d_->update(combined, num_iterations_);
            return field_2d_->get_saliency();
        }
    }

    void set_weights(const std::unordered_map<std::string, double>& weights) override {
        weights_ = weights;
    }

private:
    std::unique_ptr<NeuralField2D> field_2d_;
    std::unique_ptr<NeuralField3D> field_3d_;
    std::unordered_map<std::string, double> weights_;
    int field_size_, num_iterations_ = 50;
    double alpha_, beta_, rest_value_;

    cv::Mat combine_features(const std::vector<FeatureMap>& features);
    void setup_kernels(const Config& config);
};
```

### **4.3 Simple WTA Selector**

```cpp
class WinnerTakeAllSelector : public IAttentionSelector {
public:
    void configure(const Config& config) override {
        ior_radius_ = config.get<int>("ior_radius", 20);
        ior_strength_ = config.get<double>("ior_strength", 0.5);
        ior_decay_ = config.get<double>("ior_decay", 0.95);
    }

    cv::Point select(
        const SaliencyMap& saliency,
        const AttentionState& state
    ) override {
        cv::Mat inhibited = apply_ior(saliency.map);

        // Find global maximum
        double max_val;
        cv::Point max_loc;
        cv::minMaxLoc(inhibited, nullptr, &max_val, nullptr, &max_loc);

        // Add to IOR list
        ior_locations_.push_back({max_loc, ior_strength_});

        // Decay existing IOR
        for (auto& ior : ior_locations_) {
            ior.strength *= ior_decay_;
        }

        // Remove weak IOR
        ior_locations_.erase(
            std::remove_if(ior_locations_.begin(), ior_locations_.end(),
                          [](const auto& ior) { return ior.strength < 0.01; }),
            ior_locations_.end()
        );

        return max_loc;
    }

    void reset() override {
        ior_locations_.clear();
    }

private:
    struct IOR {
        cv::Point location;
        double strength;
    };

    std::vector<IOR> ior_locations_;
    int ior_radius_;
    double ior_strength_, ior_decay_;

    cv::Mat apply_ior(const cv::Mat& saliency);
};
```

---

## 5. **Configuration System**

### **5.1 YAML Configuration Example**

```yaml
# config/attention_pipeline.yaml

pipeline:
  name: "stereo_attention_demo"
  input:
    type: "stereo_video"
    left_path: "data/left_video.mp4"
    right_path: "data/right_video.mp4"

  features:
    - name: "color"
      plugin: "color_feature"
      weight: 1.2
      config:
        method: "opponent_colors"
        num_scales: 3

    - name: "stereo_depth"
      plugin: "stereo_depth_feature"
      weight: 0.7
      config:
        min_disparity: -25
        max_disparity: 0
        block_size: 15

    - name: "symmetry"
      plugin: "symmetry_feature"
      weight: 1.0
      config:
        num_orientations: 6
        scales: [1, 2, 4]

  integration:
    plugin: "neural_field_2d"
    config:
      field_size: 64
      alpha: 0.5
      beta: 30.0
      num_iterations: 50
      kernel_type: "difference_of_gaussians"

  selection:
    plugin: "wta_with_ior"
    config:
      ior_radius: 20
      ior_strength: 0.5
      ior_decay: 0.95

  recognition:
    - plugin: "yolo_detector"
      config:
        model: "models/yolov5s.pt"
        confidence: 0.5

  visualization:
    - plugin: "heatmap_overlay"
      config:
        colormap: "jet"
        alpha: 0.6

    - plugin: "scanpath_recorder"
      config:
        output: "results/scanpath.mp4"
        show_ior: true
```

### **5.2 Configuration Loading**

```cpp
class ConfigLoader {
public:
    static Config load_from_yaml(const std::string& path) {
        YAML::Node yaml = YAML::LoadFile(path);

        Config config;
        // Parse and validate
        config.pipeline_name = yaml["pipeline"]["name"].as<std::string>();

        // Load input configuration
        auto input = yaml["pipeline"]["input"];
        config.input_config = parse_input_config(input);

        // Load feature configurations
        for (const auto& feature : yaml["pipeline"]["features"]) {
            config.features.push_back(parse_feature_config(feature));
        }

        // ... parse remaining sections

        return config;
    }

private:
    static InputConfig parse_input_config(const YAML::Node& node);
    static FeatureConfig parse_feature_config(const YAML::Node& node);
    // ...
};
```

---

## 6. **Python Bindings with pybind11**

### **6.1 Python API**

```cpp
// python/bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Convert between cv::Mat and numpy
cv::Mat numpy_to_mat(py::array_t<uint8_t> input);
py::array_t<uint8_t> mat_to_numpy(const cv::Mat& mat);

PYBIND11_MODULE(pyattention, m) {
    m.doc() = "Visual Attention Framework Python Bindings";

    // Expose Frame
    py::class_<Frame>(m, "Frame")
        .def(py::init<>())
        .def_property("left",
            [](Frame& f) { return mat_to_numpy(f.left); },
            [](Frame& f, py::array_t<uint8_t> arr) { f.left = numpy_to_mat(arr); })
        .def_property("right",
            [](Frame& f) -> std::optional<py::array_t<uint8_t>> {
                if (f.right) return mat_to_numpy(*f.right);
                return std::nullopt;
            },
            [](Frame& f, py::array_t<uint8_t> arr) {
                f.right = numpy_to_mat(arr);
            })
        .def("is_stereo", &Frame::is_stereo);

    // Expose FeatureMap
    py::class_<FeatureMap>(m, "FeatureMap")
        .def_readonly("name", &FeatureMap::name)
        .def_property_readonly("data",
            [](const FeatureMap& fm) { return mat_to_numpy(fm.data); })
        .def_readonly("confidence", &FeatureMap::confidence);

    // Expose SaliencyMap
    py::class_<SaliencyMap>(m, "SaliencyMap")
        .def_property_readonly("map",
            [](const SaliencyMap& sm) { return mat_to_numpy(sm.map); })
        .def_readonly("peaks", &SaliencyMap::peaks)
        .def_readonly("focus", &SaliencyMap::focus);

    // Expose Pipeline
    py::class_<AttentionPipeline>(m, "AttentionPipeline")
        .def_static("from_config",
            [](const std::string& config_path) {
                auto config = ConfigLoader::load_from_yaml(config_path);
                return AttentionPipeline::Builder()
                    .set_config(config)
                    .build();
            })
        .def("process_frame", &AttentionPipeline::process_frame)
        .def("process_sequence", &AttentionPipeline::process_sequence)
        .def_property_readonly("state", &AttentionPipeline::state)
        .def("on_state_update",
            [](AttentionPipeline& p, py::function callback) {
                p.on_state_update.connect(
                    [callback](const AttentionState& state) {
                        py::gil_scoped_acquire acquire;
                        callback(state);
                    });
            });

    // Expose plugin interfaces for Python-based plugins
    py::class_<IFeatureExtractor, PyFeatureExtractor,
               std::shared_ptr<IFeatureExtractor>>(m, "IFeatureExtractor")
        .def(py::init<>())
        .def("compute", &IFeatureExtractor::compute);
}
```

### **6.2 Python Usage Example**

```python
import pyattention as pa
import numpy as np
import cv2

# Load pipeline from config
pipeline = pa.AttentionPipeline.from_config("config/attention_pipeline.yaml")

# Setup callback for visualization
def on_update(state):
    print(f"Focus: {state.saliency.focus}")

    # Visualize
    saliency_map = state.saliency.map
    cv2.imshow("Saliency", saliency_map)
    cv2.waitKey(1)

pipeline.on_state_update(on_update)

# Process video
pipeline.process_sequence(num_frames=100)

# Or process single frame
frame = pa.Frame()
frame.left = cv2.imread("test.jpg")
pipeline.process_frame()

# Access results
state = pipeline.state
print(f"Detected {len(state.features)} features")
print(f"Current focus: {state.saliency.focus}")
```

### **6.3 Python-based Feature Plugin**

```python
import pyattention as pa
import numpy as np
import cv2

class GaborFeature(pa.IFeatureExtractor):
    """Custom Gabor filter feature in Python"""

    def __init__(self):
        super().__init__()
        self.wavelength = 5.0
        self.orientations = 4

    def name(self):
        return "python_gabor"

    def version(self):
        return "1.0.0"

    def input_type(self):
        return pa.InputType.MONO

    def configure(self, config):
        self.wavelength = config.get("wavelength", 5.0)
        self.orientations = config.get("orientations", 4)

    def compute(self, frame):
        # Create Gabor kernels
        responses = []
        for theta in np.linspace(0, np.pi, self.orientations, endpoint=False):
            kernel = cv2.getGaborKernel((21, 21), 5, theta,
                                       self.wavelength, 0.5, 0)
            response = cv2.filter2D(frame.left, cv2.CV_32F, kernel)
            responses.append(response)

        # Combine responses
        combined = np.max(responses, axis=0)

        # Create feature map
        feature = pa.FeatureMap()
        feature.name = self.name()
        feature.data = combined
        feature.confidence = 1.0

        return feature

# Register plugin
pa.register_plugin(GaborFeature())
```

---

## 7. **Project Structure**

```
attention-framework/
├── CMakeLists.txt
├── conanfile.txt / vcpkg.json
├── README.md
├── LICENSE
│
├── include/
│   └── attention/
│       ├── core/
│       │   ├── frame.hpp
│       │   ├── feature_map.hpp
│       │   ├── saliency_map.hpp
│       │   └── attention_state.hpp
│       ├── interfaces/
│       │   ├── plugin.hpp
│       │   ├── feature_extractor.hpp
│       │   ├── feature_integrator.hpp
│       │   ├── attention_selector.hpp
│       │   ├── object_recognizer.hpp
│       │   └── visualizer.hpp
│       ├── pipeline/
│       │   ├── pipeline.hpp
│       │   └── builder.hpp
│       ├── plugin_manager.hpp
│       └── config.hpp
│
├── src/
│   ├── core/
│   │   └── ... (implementations)
│   ├── pipeline/
│   │   └── pipeline.cpp
│   ├── plugin_manager.cpp
│   └── config.cpp
│
├── plugins/
│   ├── features/
│   │   ├── color/
│   │   │   ├── CMakeLists.txt
│   │   │   └── color_feature.cpp
│   │   ├── stereo/
│   │   │   └── stereo_depth_feature.cpp
│   │   ├── symmetry/
│   │   ├── motion/
│   │   └── ...
│   ├── integrators/
│   │   ├── neural_field_2d/
│   │   ├── neural_field_3d/
│   │   └── weighted_sum/
│   ├── selectors/
│   │   ├── wta/
│   │   ├── ior/
│   │   └── neural_field_selector/
│   ├── recognizers/
│   │   ├── yolo_wrapper/
│   │   ├── tensorflow_wrapper/
│   │   └── ...
│   └── visualizers/
│       ├── heatmap/
│       ├── scanpath/
│       └── 3d_viewer/
│
├── python/
│   ├── setup.py
│   ├── pyattention/
│   │   ├── __init__.py
│   │   └── bindings.cpp
│   └── examples/
│       ├── simple_attention.py
│       ├── custom_feature.py
│       └── realtime_video.py
│
├── examples/
│   ├── cpp/
│   │   ├── simple_image_attention.cpp
│   │   ├── stereo_video_attention.cpp
│   │   └── custom_pipeline.cpp
│   └── configs/
│       ├── basic_mono.yaml
│       ├── stereo_depth.yaml
│       └── multi_feature.yaml
│
├── tests/
│   ├── unit/
│   │   ├── test_frame.cpp
│   │   ├── test_pipeline.cpp
│   │   └── ...
│   ├── integration/
│   │   └── test_full_pipeline.cpp
│   └── python/
│       └── test_bindings.py
│
├── tools/
│   ├── cli/
│   │   └── attention_cli.cpp  # Command-line tool
│   └── gui/
│       └── attention_viewer.cpp  # Qt/ImGui viewer
│
├── docs/
│   ├── api/
│   ├── tutorials/
│   ├── plugin_development.md
│   └── architecture.md
│
└── data/
    ├── sample_images/
    ├── sample_videos/
    └── test_configs/
```

---

## 8. **Build System (CMake)**

### **8.1 Root CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(AttentionFramework VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_TOOLS "Build CLI and GUI tools" ON)
option(BUILD_PLUGINS "Build standard plugins" ON)

# Find dependencies
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost 1.75 REQUIRED COMPONENTS
    system filesystem thread signals2)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 REQUIRED)
endif()

if(BUILD_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
endif()

# Core library
add_subdirectory(src)

# Plugins
if(BUILD_PLUGINS)
    add_subdirectory(plugins)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

# Tools
if(BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# Tests
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Examples
add_subdirectory(examples)

# Installation
install(TARGETS attention_core
        EXPORT AttentionFrameworkTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

install(DIRECTORY include/
        DESTINATION include)

# Package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    AttentionFrameworkConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(EXPORT AttentionFrameworkTargets
        FILE AttentionFrameworkTargets.cmake
        NAMESPACE AttentionFramework::
        DESTINATION lib/cmake/AttentionFramework)
```

### **8.2 Plugin CMakeLists.txt Example**

```cmake
# plugins/features/stereo/CMakeLists.txt

add_library(stereo_depth_feature MODULE
    stereo_depth_feature.cpp
    stereo_depth_feature.hpp)

target_link_libraries(stereo_depth_feature
    PRIVATE
        AttentionFramework::attention_core
        ${OpenCV_LIBS}
        Boost::boost)

set_target_properties(stereo_depth_feature PROPERTIES
    PREFIX ""  # Remove 'lib' prefix on Unix
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins)

install(TARGETS stereo_depth_feature
        LIBRARY DESTINATION lib/attention/plugins)
```

---

## 9. **Key Implementation Details**

### **9.1 Thread Pool for Parallel Features**

```cpp
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency())
        : stop_(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });

                        if (stop_ && tasks_.empty()) return;

                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) throw std::runtime_error("ThreadPool stopped");
            tasks_.emplace([task]() { (*task)(); });
        }
        condition_.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
};
```

### **9.2 Modern Neural Field Implementation**

```cpp
template<typename T = float>
class NeuralField2D {
public:
    NeuralField2D(int width, int height)
        : width_(width), height_(height),
          activity_(height, width, CV_32F),
          sigmoid_(height, width, CV_32F) {
        activity_.setTo(cv::Scalar(rest_value_));
    }

    void set_parameters(double alpha, double beta,
                       double rest_value, double global_mult) {
        alpha_ = alpha;
        beta_ = beta;
        rest_value_ = rest_value;
        global_mult_ = global_mult;
    }

    void set_kernel(const cv::Mat& kernel) {
        kernel_ = kernel.clone();
    }

    void update(const cv::Mat& input, int iterations = 50) {
        // Normalize input
        cv::Mat normed_input;
        cv::normalize(input, normed_input, 0, 1, cv::NORM_MINMAX);

        for (int i = 0; i < iterations; ++i) {
            // Compute sigmoid
            compute_sigmoid();

            // Convolve sigmoid with kernel
            cv::Mat lateral;
            cv::filter2D(sigmoid_, lateral, CV_32F, kernel_);

            // Update equation:
            // dA/dt = -alpha * A + global * lateral + input + rest
            cv::Mat delta = -alpha_ * activity_
                          + global_mult_ * lateral
                          + input_mult_ * normed_input;

            activity_ += delta;

            // Add resting value
            activity_ += cv::Scalar(rest_value_);

            // Check convergence
            if (check_convergence(delta)) break;
        }
    }

    SaliencyMap get_saliency() const {
        SaliencyMap result;

        // Normalize activity to [0, 1]
        cv::normalize(sigmoid_, result.map, 0, 1, cv::NORM_MINMAX);

        // Find local maxima
        result.peaks = find_local_maxima(result.map);

        // Find global maximum
        cv::Point max_loc;
        cv::minMaxLoc(result.map, nullptr, nullptr, nullptr, &max_loc);
        result.focus = max_loc;

        return result;
    }

private:
    int width_, height_;
    cv::Mat activity_;
    cv::Mat sigmoid_;
    cv::Mat kernel_;

    double alpha_ = 0.5;
    double beta_ = 30.0;
    double rest_value_ = -0.25;
    double global_mult_ = 1.0;
    double input_mult_ = 1.0;

    void compute_sigmoid() {
        // Approximate sigmoid: 1 / (1 + exp(-beta * x))
        sigmoid_ = 1.0 / (1.0 + cv::Mat(-beta_ * activity_).mul(-1.0));
        cv::exp(-beta_ * activity_, sigmoid_);
        cv::add(1.0, sigmoid_, sigmoid_);
        cv::divide(1.0, sigmoid_, sigmoid_);
    }

    bool check_convergence(const cv::Mat& delta) {
        double change = cv::norm(delta, cv::NORM_L1);
        return change < convergence_threshold_;
    }

    std::vector<cv::Point> find_local_maxima(const cv::Mat& map) {
        // Use non-maximum suppression
        std::vector<cv::Point> maxima;
        // ... implementation
        return maxima;
    }

    double convergence_threshold_ = 0.01;
};
```

---

## 10. **Development Roadmap**

### **Phase 1: Core Framework (2-3 months)**
1. Define all interfaces
2. Implement plugin manager
3. Implement pipeline engine
4. Configuration system
5. Basic unit tests

### **Phase 2: Essential Plugins (2 months)**
1. Input sources (image, video, camera, stereo)
2. Basic features (color, intensity, orientation)
3. Simple integrators (weighted sum)
4. WTA selector
5. Basic visualizers

### **Phase 3: Advanced Features (2 months)**
1. Stereo depth feature
2. Symmetry feature
3. Motion/optical flow feature
4. Neural field integrators (2D, 3D)
5. IOR selector

### **Phase 4: Python Integration (1 month)**
1. pybind11 bindings
2. Python examples
3. Python plugin support
4. Jupyter notebook tutorials

### **Phase 5: Tools & Integration (1-2 months)**
1. CLI tool
2. GUI viewer (Qt or ImGui)
3. 3rd party wrappers (YOLO, TensorFlow)
4. ROS2 nodes (optional)

### **Phase 6: Documentation & Polish (1 month)**
1. API documentation (Doxygen)
2. Tutorials
3. Example projects
4. Performance optimization
5. Comprehensive tests

---

## 11. **Advantages Over Original Code**

1. **Modularity**: Plugin architecture allows easy extension
2. **Type Safety**: Modern C++ with proper interfaces
3. **Memory Safety**: Smart pointers, RAII
4. **Testability**: Dependency injection, mocking
5. **Performance**: OpenMP/TBB parallelization, efficient OpenCV
6. **Portability**: CMake, cross-platform
7. **Usability**: Python bindings, configuration files
8. **Maintainability**: Clear architecture, documentation
9. **Extensibility**: Easy to add features, integrators, selectors
10. **Integration**: Works with modern tools (YOLO, TensorFlow, ROS)

---

## 12. **Example End-to-End Workflow**

```bash
# Build
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make -j8

# Run CLI tool
./attention_cli \
    --config ../configs/stereo_attention.yaml \
    --input ../data/stereo_video.mp4 \
    --output results/

# Python script
python3 << EOF
import pyattention as pa

# Load and run pipeline
pipeline = pa.AttentionPipeline.from_config("configs/stereo_attention.yaml")

# Add custom callback
def on_focus(state):
    print(f"Attended to: {state.saliency.focus}")

pipeline.on_attention_shift(on_focus)
pipeline.process_sequence(100)
EOF

# Develop custom plugin
cd plugins/features/myfeature
# ... implement IFeatureExtractor
cmake --build .
# Plugin auto-discovered at runtime
```

This architecture provides a **solid foundation** for modern visual attention research while maintaining the scientific insights from the original PhD work.
