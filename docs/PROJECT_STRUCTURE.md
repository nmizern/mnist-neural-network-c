# Neural Network in C - Project Structure

## Directory Layout

```
nn/
├── CMakeLists.txt              # Build configuration
├── .gitlab-ci.yml              # GitLab CI/CD pipeline
├── .gitignore
│
├── include/                    # Header files
│   ├── neuralnet.h             # Main header (includes all modules)
│   └── nn/
│       ├── matrix.h            # Matrix structure and operations
│       ├── nn_functions.h      # Activation functions + loss
│       ├── network.h           # Network structure, training, evaluation
│       └── mnist.h             # MNIST dataset loading (PNG via libpng)
│
├── src/                        # Implementation files
│   ├── matrix.c
│   ├── nn_functions.c
│   ├── network.c
│   └── mnist.c
│
├── examples/
│   ├── CMakeLists.txt
│   └── mnist_train.c           # Full MNIST training pipeline
│
├── data/                       # Dataset directory (gitignored)
│   └── mnist-pngs/             # https://github.com/rasbt/mnist-pngs
│       ├── train/0..9/
│       └── test/0..9/
│
└── docs/
    └── PROJECT_STRUCTURE.md    # This file
```

## Build

Prerequisites: C compiler (GCC), CMake 3.12+, libpng

```bash
mkdir build && cd build
cmake ..
cmake --build .
./examples/mnist_train ../data/mnist-pngs
```
