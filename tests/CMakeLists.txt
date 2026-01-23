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
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
