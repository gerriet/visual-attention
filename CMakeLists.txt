cmake_minimum_required(VERSION 3.20)
project(AttentionFramework VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to an optimized build: a plain `cmake -S . -B build` otherwise leaves
# CMAKE_BUILD_TYPE empty (no -O flags at all), and every study and timing then
# runs several times slower than it needs to. Multi-config generators (Xcode,
# Visual Studio) choose at build time and are left alone.
get_property(ATTENTION_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT ATTENTION_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type given - defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# Find dependencies. OpenCV 4 is pinned (the code predates the 5.0 API); once
# Homebrew's default `opencv` moved to 5.x, `brew install opencv@4` became the
# supported route, and that formula is keg-only — hence the explicit prefix.
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/opencv@4")
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
    src/features/munsell_color_feature.cpp
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
    src/fusion/priority_map.cpp
    src/selection/selection.cpp
    src/selection/neural_field_selection.cpp
    src/selection/neural_field_3d.cpp
    src/selection/kalman_mot_selection.cpp
    src/selection/normalization_selection.cpp
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

# Bake the linked OpenCV's own haarcascades directory into the build, so the
# haar-face processor finds its cascade without $ATTENTION_HAAR_DIR even when
# OpenCV is keg-only or versioned (e.g. Homebrew opencv@4) and therefore on
# none of the fixed search paths. The unresolved OpenCV_DIR prefix comes first:
# OpenCV_INSTALL_PATH is symlink-resolved (a versioned Homebrew Cellar path
# that a `brew upgrade` + cleanup deletes), the opt/ symlink stays valid.
get_filename_component(_opencv_prefix "${OpenCV_DIR}/../../.." ABSOLUTE)
set(ATTENTION_OPENCV_HAAR_DIR "")
foreach(_haar_dir "${_opencv_prefix}/share/opencv4/haarcascades"
                  "${OpenCV_INSTALL_PATH}/share/opencv4/haarcascades"
                  "${OpenCV_INSTALL_PATH}/share/opencv/haarcascades")
    if(NOT ATTENTION_OPENCV_HAAR_DIR AND EXISTS "${_haar_dir}/haarcascade_frontalface_default.xml")
        set(ATTENTION_OPENCV_HAAR_DIR "${_haar_dir}")
    endif()
endforeach()
if(ATTENTION_OPENCV_HAAR_DIR)
    target_compile_definitions(attention_core PRIVATE ATTENTION_OPENCV_HAAR_DIR="${ATTENTION_OPENCV_HAAR_DIR}")
    message(STATUS "Haar cascades: ${ATTENTION_OPENCV_HAAR_DIR}")
else()
    message(STATUS "Haar cascades: not found under OpenCV (haar-face needs $ATTENTION_HAAR_DIR)")
endif()

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
