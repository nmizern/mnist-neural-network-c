# Neural Network in C

A fully-connected neural network library implemented in C for MNIST digit classification.

**Authors:** Mikita Mizerkin, Idirene Daris
**Course:** TEI S7 - Neural Networks Project

## Project Goal

Implement a neural network from scratch in C that can:
1. Create networks with variable number of layers and sizes
2. Perform inference (forward propagation)
3. Train using backpropagation and SGD
4. Classify MNIST handwritten digits

## Quick Start

### Prerequisites

- C compiler (GCC, Clang, or MSVC)
- CMake 3.12+
- libpng (for PNG image loading)

### Install libpng

```bash
# Ubuntu/Debian
sudo apt-get install libpng-dev

# macOS
brew install libpng

# Windows (vcpkg)
vcpkg install libpng
```

### Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Run Tests

```bash
ctest --output-on-failure
```

### Train on MNIST

1. Clone MNIST PNG dataset:
   ```bash
   git clone https://github.com/rasbt/mnist-pngs data/mnist-pngs
   ```

2. Run training:
   ```bash
   ./examples/mnist_train ../data/mnist-pngs
   ```

## Project Structure

```
nn/
├── include/
│   ├── neuralnet.h         # Main API header
│   └── nn/                 # Module headers
│       ├── matrix.h        # Matrix operations
│       ├── activation.h    # ReLU, Sigmoid
│       ├── layer.h         # Layer structure
│       ├── network.h       # Network structure
│       ├── loss.h          # MSE loss
│       ├── optimizer.h     # SGD + backpropagation
│       ├── mnist.h         # MNIST loading (PNG)
│       └── metrics.h       # Accuracy metrics
├── src/                    # Implementations
├── tests/                  # Unit tests
├── examples/               # Usage examples
└── docs/                   # Documentation
```

## Documentation

- [Project Structure](docs/PROJECT_STRUCTURE.md) - Detailed structure description
- [TODO](docs/TODO.md) - Implementation checklist

## Dependencies

- **Standard C libraries** (stdio.h, stdlib.h, math.h, etc.)
- **libpng** (png.h) - for loading PNG images

## License

University project - TEI S7
