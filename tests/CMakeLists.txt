cmake_minimum_required(VERSION 3.12)

# Test files
file(GLOB TEST_SOURCES "test_*.c")

foreach(test_file ${TEST_SOURCES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} neuralnet)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
