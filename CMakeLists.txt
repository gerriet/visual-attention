cmake_minimum_required(VERSION 3.20)
project(AttentionFramework VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find dependencies
find_package(OpenCV 4 REQUIRED)
# find_package(Eigen3 REQUIRED)  # Uncomment when needed

# yaml-cpp (homebrew installation)
set(YAML_CPP_DIR "/opt/homebrew/opt/yaml-cpp")
include_directories(${YAML_CPP_DIR}/include)
link_directories(${YAML_CPP_DIR}/lib)

# Include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Build main executable
add_executable(attention
    src/main.cpp
    src/pipeline/attention_pipeline.cpp
    src/features/color_feature.cpp
    src/features/intensity_feature.cpp
    src/features/symmetry_feature.cpp
    src/config/config_loader.cpp
    src/visualization/visualizer.cpp
)
target_link_libraries(attention ${OpenCV_LIBS} yaml-cpp)

# Optional: clang-format targets for code formatting
find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
    # Get all C++ source files
    file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/include/*.h
        ${CMAKE_SOURCE_DIR}/src/*.cpp
    )

    # Target to check if formatting is needed (CI/CD)
    add_custom_target(format-check
        COMMAND ${CLANG_FORMAT} --dry-run --Werror ${ALL_CXX_SOURCE_FILES}
        COMMENT "Checking code formatting..."
        VERBATIM
    )

    # Target to apply formatting
    add_custom_target(format
        COMMAND ${CLANG_FORMAT} -i ${ALL_CXX_SOURCE_FILES}
        COMMENT "Formatting code with clang-format..."
        VERBATIM
    )

    message(STATUS "clang-format found: ${CLANG_FORMAT}")
    message(STATUS "  Run 'cmake --build . --target format' to format code")
    message(STATUS "  Run 'cmake --build . --target format-check' to check formatting (CI mode)")
    message(STATUS "  Or use 'ninja format' / 'make format' depending on your generator")
else()
    message(WARNING "clang-format not found. Code formatting targets unavailable.")
endif()

message(STATUS "")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "")
message(STATUS "Build the project with: cmake --build .")
message(STATUS "Run with: ./attention <image_path>")
