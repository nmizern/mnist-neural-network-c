# Neural Network in C - Project Structure

## Directory Layout

```
nn/
├── CMakeLists.txt              # Main CMake configuration
├── .gitlab-ci.yml              # GitLab CI/CD pipeline
├── .gitignore                  # Git ignore rules
│
├── include/                    # Header files
│   ├── neuralnet.h             # Main public API (includes all modules)
│   ├── nn/                     # Module headers
│   │   ├── matrix.h            # Matrix/vector operations
│   │   ├── activation.h        # Activation functions (ReLU, Sigmoid)
│   │   ├── layer.h             # Single layer structure
│   │   ├── network.h           # Network structure and operations
│   │   ├── loss.h              # Loss functions (MSE)
│   │   ├── optimizer.h         # Training algorithms (SGD, backprop)
│   │   ├── mnist.h             # MNIST dataset loading
│   │   └── metrics.h           # Evaluation metrics (accuracy)
│   └── third_party/            # Third-party headers
│       └── stb_image.h         # PNG loading (download separately)
│
├── src/                        # Source implementation files
│   ├── matrix.c                # Matrix operations
│   ├── activation.c            # Activation functions
│   ├── layer.c                 # Layer operations
│   ├── network.c               # Network operations
│   ├── loss.c                  # Loss functions
│   ├── optimizer.c             # Training (SGD + backpropagation)
│   ├── mnist.c                 # MNIST loading
│   └── metrics.c               # Accuracy calculation
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
├── data/                       # Dataset directory (gitignored)
│   └── mnist-pngs/             # MNIST PNG images
│
└── docs/                       # Documentation
    ├── PROJECT_STRUCTURE.md    # This file
    └── TODO.md                 # Implementation checklist
```

## Module Descriptions

### Core Modules (in implementation order)

| Module | File | Description |
|--------|------|-------------|
| **Matrix** | `matrix.h/c` | Fundamental matrix/vector operations. Must implement first. |
| **Activation** | `activation.h/c` | ReLU, Sigmoid and their derivatives for backprop. |
| **Layer** | `layer.h/c` | Single fully-connected layer with weights and biases. |
| **Network** | `network.h/c` | Complete network as sequence of layers. |
| **Loss** | `loss.h/c` | MSE loss function and one-hot encoding. |
| **Optimizer** | `optimizer.h/c` | **KEY MODULE**: SGD and backpropagation algorithm. |
| **MNIST** | `mnist.h/c` | Load MNIST PNG images into memory. |
| **Metrics** | `metrics.h/c` | Calculate classification accuracy. |

### Recommended Implementation Order

1. `matrix.c` - Foundation for everything
2. `activation.c` - Simple, independent functions
3. `layer.c` - Depends on matrix and activation
4. `network.c` - Depends on layer
5. `loss.c` - Independent, needed for training
6. `optimizer.c` - Most complex, depends on all above
7. `mnist.c` - Can be done in parallel
8. `metrics.c` - Simple, depends on network

## Build Instructions

### Prerequisites

- C compiler (GCC, Clang, or MSVC)
- CMake 3.12+
- stb_image.h (download from GitHub)

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
