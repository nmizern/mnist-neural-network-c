# Tests CMakeLists.txt

# Automatically find all test source files
file(GLOB TEST_SOURCES "test_*.c")

foreach(test_file ${TEST_SOURCES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} neuralnet)
    # Link math library on Unix
    if(UNIX)
        target_link_libraries(${test_name} m)
    endif()
    if(NOT ${test_name} STREQUAL "test_interactive")
        add_test(NAME ${test_name} COMMAND ${test_name})
    endif()

    # Copy required DLLs next to test executable (Windows + dynamic libs)
    if(WIN32 AND PNG_LIBRARIES)
        add_custom_command(TARGET ${test_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:PNG::PNG>
                $<TARGET_FILE_DIR:${test_name}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:ZLIB::ZLIB>
                $<TARGET_FILE_DIR:${test_name}>
        )
    endif()
endforeach()
