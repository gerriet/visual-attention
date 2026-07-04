# Test suite: characterization (golden map) tests in C++ via Catch2, plus
# behavioral scanpath tests that exercise the CLI + interchange format and
# compare with the Python evaluation layer.

find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.5.4.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(Catch2)
endif()

add_executable(characterization_tests test_characterization.cpp test_config.cpp test_selection.cpp)
target_link_libraries(characterization_tests PRIVATE attention_core Catch2::Catch2WithMain)
target_compile_definitions(characterization_tests PRIVATE
    ATTENTION_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_test(NAME characterization COMMAND characterization_tests)

# Behavioral tests: run the CLI with --emit-json, then compare the emitted
# scanpath against the golden result with loose tolerances (see V2_ROADMAP.md:
# the replication bar is "mainly equivalent", not exact).
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    set(BEHAVIORAL_IMAGES input inputc lena)
    foreach(img IN LISTS BEHAVIORAL_IMAGES)
        add_test(NAME emit_json_${img}
            COMMAND attention ${CMAKE_SOURCE_DIR}/data/test_images/${img}.png
                    --no-display --emit-json ${CMAKE_BINARY_DIR}/behavioral/${img}.json
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
        set_tests_properties(emit_json_${img} PROPERTIES FIXTURES_SETUP ${img}_json)

        add_test(NAME scanpath_${img}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/eval/compare_scanpaths.py
                    ${CMAKE_SOURCE_DIR}/tests/golden/${img}/result.json
                    ${CMAKE_BINARY_DIR}/behavioral/${img}.json
                    --pos-tol 20 --order-tol 1)
        set_tests_properties(scanpath_${img} PROPERTIES FIXTURES_REQUIRED ${img}_json)
    endforeach()
else()
    message(WARNING "Python3 not found - behavioral scanpath tests disabled")
endif()
