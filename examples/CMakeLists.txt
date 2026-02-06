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

    # Copy required DLLs next to executable after build (Windows + dynamic libs)
    if(WIN32 AND PNG_LIBRARIES)
        add_custom_command(TARGET ${example_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:PNG::PNG>
                $<TARGET_FILE_DIR:${example_name}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:ZLIB::ZLIB>
                $<TARGET_FILE_DIR:${example_name}>
            COMMENT "Copying libpng/zlib DLLs for ${example_name}"
        )
    endif()
endforeach()
