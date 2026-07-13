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

add_executable(characterization_tests test_characterization.cpp test_config.cpp test_selection.cpp test_stereo_onset.cpp test_system.cpp test_processors.cpp test_alt_features.cpp test_kalman_mot.cpp test_normalization.cpp)
target_link_libraries(characterization_tests PRIVATE attention_core Catch2::Catch2WithMain)
target_compile_definitions(characterization_tests PRIVATE
    ATTENTION_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_test(NAME characterization COMMAND characterization_tests)

# CLI contract: every binary answers --help with a usage synopsis, and the
# main CLI rejects unknown flags instead of treating them as an image path.
foreach(bin attention pipeline_test debug_color)
    add_test(NAME help_${bin} COMMAND ${bin} --help)
    set_tests_properties(help_${bin} PROPERTIES PASS_REGULAR_EXPRESSION "Usage")
endforeach()
add_test(NAME unknown_flag_rejected COMMAND attention --no-such-flag)
set_tests_properties(unknown_flag_rejected PROPERTIES
    WILL_FAIL TRUE)

# M8 live demonstrator, headless: annotated frames + object-file plugins over
# the synthetic motion sequence ("Live complete" prints only after the loop).
add_test(NAME live_headless_smoke
    COMMAND attention --live ${CMAKE_SOURCE_DIR}/data/test_images/motion_seq
            --no-display --frames 3 --output ${CMAKE_BINARY_DIR}/live_smoke
            --processors roi-probe,region-descriptor
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(live_headless_smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "Live complete: 3 frames")

# M9 alternative saliency features: the config-driven plug-in path runs the
# non-thesis operators (spectral-residual, phase-spectrum, frequency-tuned,
# image-signature, boolean-map, minimum-barrier) end to end, standalone.
# Guards the registry keys and configs/alternative.yaml.
add_test(NAME alt_features_profile
    COMMAND attention ${CMAKE_SOURCE_DIR}/data/test_images/inputc.png
            --config ${CMAKE_SOURCE_DIR}/configs/alternative.yaml --no-display
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(alt_features_profile PROPERTIES
    PASS_REGULAR_EXPRESSION "Features extracted: 6")

# kalman-mot selection backend (docs/SELECTION_BACKENDS.md): detection + Kalman
# multi-object tracking as an alternative to the neural field, exercised across
# frames of the motion sequence via the live path.
add_test(NAME kalman_mot_stream
    COMMAND attention --live ${CMAKE_SOURCE_DIR}/data/test_images/motion_seq
            --config ${CMAKE_SOURCE_DIR}/configs/kalman.yaml
            --no-display --frames 3 --output ${CMAKE_BINARY_DIR}/kalman_smoke
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(kalman_mot_stream PROPERTIES
    PASS_REGULAR_EXPRESSION "Live complete: 3 frames")

# ...and that kalman-mot actually emits peaks through the config path (a still,
# deterministic): guards against a regression to silently-empty selection that
# the streaming completion banner above cannot detect.
add_test(NAME kalman_mot_peaks
    COMMAND attention ${CMAKE_SOURCE_DIR}/data/test_images/inputc.png
            --config ${CMAKE_SOURCE_DIR}/configs/kalman.yaml --no-display
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(kalman_mot_peaks PROPERTIES
    PASS_REGULAR_EXPRESSION "Peaks detected: [1-9]")

# normalization selection backend (docs/SELECTION_BACKENDS.md, option D):
# divisive-normalization competition, emitting peaks through the config path.
add_test(NAME normalization_peaks
    COMMAND attention ${CMAKE_SOURCE_DIR}/data/test_images/inputc.png
            --config ${CMAKE_SOURCE_DIR}/configs/normalization.yaml --no-display
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(normalization_peaks PROPERTIES
    PASS_REGULAR_EXPRESSION "Peaks detected: [1-9]")

# Behavioral tests: run the CLI with --emit-json, then compare the emitted
# scanpath against the golden result with loose tolerances (see V2_ROADMAP.md:
# the replication bar is "mainly equivalent", not exact).
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    set(BEHAVIORAL_IMAGES input inputc)
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

        # Modern-track benchmark (M7): the C++ thesis model vs the Python modern
        # models, cross-model, over two thesis images. Exercises the full path
        # (model runners → interchange → aggregate report).
        add_test(NAME eval_benchmark_smoke
            COMMAND ${EVAL_PYTHON} -m attention_eval.benchmark
                    --images ${CMAKE_SOURCE_DIR}/data/test_images/input.png
                             ${CMAKE_SOURCE_DIR}/data/test_images/inputc.png
                    --out ${CMAKE_BINARY_DIR}/benchmark_smoke
                    --model cpp:$<TARGET_FILE:attention>:${CMAKE_SOURCE_DIR}/configs/thesis.yaml=thesis
                    --model spectral-residual --model center-bias --reference thesis
                    --report ${CMAKE_BINARY_DIR}/benchmark_smoke/report.md
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/eval)
    else()
        message(STATUS "eval/.venv not found - Python eval tests disabled")
    endif()

    # Stereo profile (disparity feature + 3D neural-field selection) behavioral
    # golden, on the deterministic synthetic stereo pair (M5).
    add_test(NAME emit_json_stereo
        COMMAND attention --stereo
                ${CMAKE_SOURCE_DIR}/data/test_images/stereo/left.png
                ${CMAKE_SOURCE_DIR}/data/test_images/stereo/right.png
                --config ${CMAKE_SOURCE_DIR}/configs/stereo.yaml
                --emit-json ${CMAKE_BINARY_DIR}/behavioral/stereo.json
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(emit_json_stereo PROPERTIES FIXTURES_SETUP stereo_json)

    add_test(NAME scanpath_stereo
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/eval/compare_scanpaths.py
                ${CMAKE_SOURCE_DIR}/tests/golden/stereo/result.json
                ${CMAKE_BINARY_DIR}/behavioral/stereo.json
                --pos-tol 20 --order-tol 1)
    set_tests_properties(scanpath_stereo PROPERTIES FIXTURES_REQUIRED stereo_json)

    # AttentionSystem (M6): second stage + Exploration behavior over the
    # synthetic motion sequence, compared by focus position per frame.
    add_test(NAME emit_scanpath_attend
        COMMAND attention --attend ${CMAKE_SOURCE_DIR}/data/test_images/motion_seq
                --output ${CMAKE_BINARY_DIR}/attend
                --emit-scanpath ${CMAKE_BINARY_DIR}/attend/scanpath.json
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(emit_scanpath_attend PROPERTIES FIXTURES_SETUP attend_scanpath)

    add_test(NAME scanpath_attend
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/eval/compare_scanpath_json.py
                ${CMAKE_SOURCE_DIR}/tests/golden/attend/scanpath.json
                ${CMAKE_BINARY_DIR}/attend/scanpath.json
                --pos-tol 20)
    set_tests_properties(scanpath_attend PROPERTIES FIXTURES_REQUIRED attend_scanpath)

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
