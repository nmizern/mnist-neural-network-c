# Neural Network in C

A fully-connected neural network library implemented in pure C for MNIST digit classification.

**Authors:** Mikita Mizerkin, Idirene Daris
**Course:** TEI S7 - Neural Networks Project

## Project Goal

Implement a neural network from scratch in C (using only standard libraries) that can:
1. Create networks with variable number of layers and sizes
2. Perform inference (forward propagation)
3. Train using backpropagation and SGD
4. Classify MNIST handwritten digits

## Quick Start

### Prerequisites

- C compiler (GCC, Clang, or MSVC)
- CMake 3.12+

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

1. Download MNIST PNG dataset:
   ```bash
   git clone https://github.com/rasbt/mnist-pngs data/mnist-pngs
   ```

2. Download stb_image.h:
   ```bash
   curl -o include/third_party/stb_image.h \
     https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
   ```

3. Run training:
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
│       ├── mnist.h         # MNIST loading
│       └── metrics.h       # Accuracy metrics
├── src/                    # Implementations
├── tests/                  # Unit tests
├── examples/               # Usage examples
└── docs/                   # Documentation
```

## Documentation

- [Project Structure](docs/PROJECT_STRUCTURE.md) - Detailed structure description
- [TODO](docs/TODO.md) - Implementation checklist

## License

University project - TEI S7
