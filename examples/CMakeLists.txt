# Examples CMakeLists.txt

# Automatically find all example source files
file(GLOB EXAMPLE_SOURCES "*.c")

foreach(example_file ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_file} NAME_WE)
    add_executable(${example_name} ${example_file})
    target_link_libraries(${example_name} neuralnet)
    # Link math library on Unix
    if(UNIX)
        target_link_libraries(${example_name} m)
    endif()
endforeach()
