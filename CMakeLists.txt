cmake_minimum_required(VERSION 3.20)
project(AttentionFramework VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find dependencies
find_package(OpenCV 4 REQUIRED)
# find_package(Eigen3 REQUIRED)  # Uncomment when needed
# find_package(yaml-cpp REQUIRED)  # Uncomment when needed

# Include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Placeholder for now - will add sources in Phase 1
# add_subdirectory(src)
# add_subdirectory(examples)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "")
message(STATUS "Project structure created. Ready to start Phase 1!")
message(STATUS "See docs/PHASE1_ACTION_PLAN.md for next steps.")
