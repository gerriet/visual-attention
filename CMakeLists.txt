cmake_minimum_required(VERSION 3.20)
project(AttentionFramework VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find dependencies
find_package(OpenCV 4 REQUIRED)
find_package(OpenMP)

# yaml-cpp: prefer an installed CMake package config; the Homebrew prefix is
# appended so a plain `brew install yaml-cpp` is found without configuration
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/yaml-cpp")
find_package(yaml-cpp CONFIG REQUIRED)
if(TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
    set(YAML_CPP_TARGET yaml-cpp)
endif()

# Core library: everything except main(), shared by the CLI, examples, tests
add_library(attention_core STATIC
    src/core/frame.cpp
    src/pipeline/attention_pipeline.cpp
    src/pipeline/frame_source.cpp
    src/features/color_feature.cpp
    src/features/intensity_feature.cpp
    src/features/orientation_feature.cpp
    src/features/eccentricity_feature.cpp
    src/features/symmetry_feature.cpp
    src/features/stereo_feature.cpp
    src/features/onset_feature.cpp
    src/features/spectral_residual_feature.cpp
    src/features/frequency_tuned_feature.cpp
    src/features/boolean_map_feature.cpp
    src/features/image_signature_feature.cpp
    src/features/phase_spectrum_feature.cpp
    src/features/minimum_barrier_feature.cpp
    src/features/debug_context.cpp
    src/features/feature_registry.cpp
    src/fusion/fusion.cpp
    src/selection/selection.cpp
    src/selection/neural_field_selection.cpp
    src/selection/neural_field_3d.cpp
    src/selection/kalman_mot_selection.cpp
    src/system/object_file.cpp
    src/system/behavior.cpp
    src/system/attention_system.cpp
    src/system/processor.cpp
    src/system/live_demonstrator.cpp
    src/config/config_loader.cpp
    src/visualization/visualizer.cpp
    src/io/result_writer.cpp
    src/io/scanpath_writer.cpp
)
target_include_directories(attention_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(attention_core SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(attention_core PUBLIC ${OpenCV_LIBS} ${YAML_CPP_TARGET})

if(OpenMP_CXX_FOUND)
    target_link_libraries(attention_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for parallel processing")
else()
    message(WARNING "OpenMP not found - symmetry feature will run sequentially")
endif()

# Main executable
add_executable(attention src/main.cpp)
target_link_libraries(attention PRIVATE attention_core)

# Examples (built by default so they cannot silently rot)
add_executable(pipeline_test examples/pipeline_test.cpp)
target_link_libraries(pipeline_test PRIVATE attention_core)

add_executable(debug_color examples/debug_color.cpp)
target_link_libraries(debug_color PRIVATE attention_core)

# Tests
include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Optional: clang-format targets for code formatting
find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
    file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/include/*.h
        ${CMAKE_SOURCE_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/examples/*.cpp
        ${CMAKE_SOURCE_DIR}/tests/*.cpp
    )

    add_custom_target(format-check
        COMMAND ${CLANG_FORMAT} --dry-run --Werror ${ALL_CXX_SOURCE_FILES}
        COMMENT "Checking code formatting..."
        VERBATIM
    )

    add_custom_target(format
        COMMAND ${CLANG_FORMAT} -i ${ALL_CXX_SOURCE_FILES}
        COMMENT "Formatting code with clang-format..."
        VERBATIM
    )

    message(STATUS "clang-format found: ${CLANG_FORMAT}")
else()
    message(WARNING "clang-format not found. Code formatting targets unavailable.")
endif()

message(STATUS "")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Build the project with: cmake --build .")
message(STATUS "Run the tests with:    ctest --test-dir build")
message(STATUS "")
