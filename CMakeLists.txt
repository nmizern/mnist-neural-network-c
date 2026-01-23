cmake_minimum_required(VERSION 3.12)
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

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files - all modules
set(SOURCES
    src/matrix.c
    src/activation.c
    src/layer.c
    src/network.c
    src/loss.c
    src/optimizer.c
    src/mnist.c
    src/metrics.c
)

# Header files
set(HEADERS
    include/neuralnet.h
    include/nn/matrix.h
    include/nn/activation.h
    include/nn/layer.h
    include/nn/network.h
    include/nn/loss.h
    include/nn/optimizer.h
    include/nn/mnist.h
    include/nn/metrics.h
)

# Create static library
add_library(neuralnet STATIC ${SOURCES} ${HEADERS})

# Link math library on Unix
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
