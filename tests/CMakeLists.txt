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

    # Python eval-layer unit tests + report smoke test (need the eval venv:
    # python3 -m venv eval/.venv && eval/.venv/bin/pip install -r eval/requirements.txt)
    set(EVAL_PYTHON ${CMAKE_SOURCE_DIR}/eval/.venv/bin/python)
    if(EXISTS ${EVAL_PYTHON})
        add_test(NAME eval_unit
            COMMAND ${EVAL_PYTHON} -m unittest discover -s tests
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/eval)
        add_test(NAME eval_report_smoke
            COMMAND ${EVAL_PYTHON} -m attention_eval.report
                    ${CMAKE_SOURCE_DIR}/tests/golden/inputc/result.json
                    ${CMAKE_SOURCE_DIR}/tests/golden/thesis_inputc/result.json
                    --names modern-default thesis-profile
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/eval)
    else()
        message(STATUS "eval/.venv not found - Python eval tests disabled")
    endif()

    # Thesis profile (neural-field selection) behavioral golden
    add_test(NAME emit_json_thesis_inputc
        COMMAND attention --config ${CMAKE_SOURCE_DIR}/configs/thesis.yaml
                ${CMAKE_SOURCE_DIR}/data/test_images/inputc.png
                --no-display --emit-json ${CMAKE_BINARY_DIR}/behavioral/thesis_inputc.json
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(emit_json_thesis_inputc PROPERTIES FIXTURES_SETUP thesis_inputc_json)

    add_test(NAME scanpath_thesis_inputc
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/eval/compare_scanpaths.py
                ${CMAKE_SOURCE_DIR}/tests/golden/thesis_inputc/result.json
                ${CMAKE_BINARY_DIR}/behavioral/thesis_inputc.json
                --pos-tol 20 --order-tol 1)
    set_tests_properties(scanpath_thesis_inputc PROPERTIES FIXTURES_REQUIRED thesis_inputc_json)
else()
    message(WARNING "Python3 not found - behavioral scanpath tests disabled")
endif()
