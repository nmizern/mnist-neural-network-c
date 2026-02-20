cmake_minimum_required(VERSION 3.12)

# Disable vcpkg's automatic DLL copy (uses powershell, breaks MinGW)
set(VCPKG_APPLOCAL_DEPS OFF CACHE BOOL "" FORCE)

project(NeuralNetwork C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler warnings for better code quality
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Find libpng and zlib
find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)

# On MSVC, find dirent compatibility header (not built-in on Windows)
if(MSVC)
    find_path(DIRENT_INCLUDE_DIR "dirent.h")
    if(NOT DIRENT_INCLUDE_DIR)
        message(FATAL_ERROR "dirent.h not found. Run: vcpkg install dirent:x64-windows")
    endif()
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PNG_INCLUDE_DIRS})

# Source files - all modules
set(SOURCES
    src/matrix.c
    src/nn_functions.c
    src/network.c
    src/mnist.c
)

# Header files
set(HEADERS
    include/neuralnet.h
    include/nn/matrix.h
    include/nn/nn_functions.h
    include/nn/network.h
    include/nn/mnist.h
)

# Create static library
add_library(neuralnet STATIC ${SOURCES} ${HEADERS})

# Link libraries
target_include_directories(neuralnet PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${PNG_INCLUDE_DIRS}
)
if(MSVC AND DIRENT_INCLUDE_DIR)
    target_include_directories(neuralnet PRIVATE ${DIRENT_INCLUDE_DIR})
endif()
target_link_libraries(neuralnet PRIVATE PNG::PNG ZLIB::ZLIB)
if(UNIX)
    target_link_libraries(neuralnet m)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS neuralnet DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

