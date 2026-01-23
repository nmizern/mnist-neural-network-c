# Neural Network in C - Project Structure

## Directory Layout

```
nn/
├── CMakeLists.txt              # Main CMake configuration
├── README.md                   # Project readme
├── TEI-S7-NN.pdf               # Reference documentation
├── .gitlab-ci.yml              # GitLab CI/CD pipeline
├── .gitignore                  # Git ignore rules
│
├── include/                    # Header files
│   ├── neuralnet.h             # Main public API (includes all modules)
│   ├── nn/                     # Module headers
│   │   ├── matrix.h            # Matrix/vector operations
│   │   ├── network.h           # Network structure and operations
│   │   ├── nn_functions.h      # Activation functions (ReLU, Sigmoid)
│   │   └── mnist.h             # MNIST dataset loading
│   └── third_party/            # Third-party headers
│       └── .gitkeep            # Placeholder (stb_image.h to be downloaded)
│
├── src/                        # Source implementation files
│   ├── matrix.c                # Matrix operations
│   ├── network.c               # Network operations
│   ├── nn_functions.c          # Activation functions
│   └── mnist.c                 # MNIST loading
│
├── tests/                      # Unit tests
│   ├── CMakeLists.txt          # Test configuration
│   ├── test_matrix.c           # Matrix operation tests
│   ├── test_activation.c       # Activation function tests
│   └── test_network.c          # Network tests
│
├── examples/                   # Usage examples
│   ├── CMakeLists.txt          # Examples configuration
│   ├── xor_example.c           # XOR problem (simple test)
│   └── mnist_train.c           # Full MNIST training
│
├── build/                      # Build directory (generated)
│
├── clean_pngtest/              # PNG test files
│   └── PNG/
│
├── data/                       # Dataset directory (gitignored)
│   └── mnist-pngs/             # MNIST PNG images
│
└── docs/                       # Documentation
    ├── PROJECT_STRUCTURE.md    # This file
    └── TODO.md                 # Implementation checklist
```

## Module Descriptions

### Core Modules (in implementation order)

| Module | Header | Source | Description |
|--------|--------|--------|-------------|
| **Matrix** | `nn/matrix.h` | `matrix.c` | Fundamental matrix/vector operations. Must implement first. |
| **NN Functions** | `nn/nn_functions.h` | `nn_functions.c` | Activation functions: ReLU, Sigmoid and their derivatives. |
| **Network** | `nn/network.h` | `network.c` | Complete network structure with layers, forward/backward pass. |
| **MNIST** | `nn/mnist.h` | `mnist.c` | Load MNIST PNG images into memory. |

### Recommended Implementation Order

1. `matrix.c` - Foundation for everything (Matrix and Vector structures)
2. `nn_functions.c` - Activation functions (Sigmoid, ReLU)
3. `network.c` - Network creation, forward pass, backpropagation
4. `mnist.c` - Dataset loading (can be done in parallel)

## Build Instructions

### Prerequisites

- C compiler (GCC, Clang, or MSVC)
- CMake 3.12+
- stb_image.h (download from GitHub, place in `include/third_party/`)

### Build Steps

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
cmake --build .

# Run tests
ctest --output-on-failure

# Run examples
./examples/xor_example
./examples/mnist_train ../data/mnist-pngs
```

### Download stb_image.h

```bash
# From project root
curl -o include/third_party/stb_image.h \
  https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
```

## Data Setup

Download MNIST PNG dataset:

```bash
git clone https://github.com/rasbt/mnist-pngs data/mnist-pngs
```

Or place it anywhere and pass the path to `mnist_train`.
