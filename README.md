[**English**](./README.md) | [Français](./README.fr.md)

# Neural Network in C

A C library for dense (fully connected) neural networks for MNIST classification, with a web interface written in Go.

**Authors:** Mikita Mizerkin, Idirene Daris

**Course:** TEI S7 - Neural Networks Project

> **Successfully presented to the examination committee — grade: 19/20, the best result in the L3 E3A class.**
>
> Developed over 10 four-hour sessions, supplemented by independent work at home.

![Web interface screenshot](./docs/image.png)

## Quick Start

The repository includes a trained model, so you can try the web interface without training the network first. Go 1.25 or later is required.

```bash
cd web
go mod download
go run . -model ../examples/mnist-pngs/mnist_model.bin -addr :4343
```

Open `http://localhost:4343` in a browser and draw a digit on the canvas.

## Results

The bundled model uses a `784 → 256 → 128 → 10` architecture. The project report records the following MNIST test results after five training epochs:

| Architecture | Test accuracy | Time per epoch |
|--------------|---------------|----------------|
| `784 → 128 → 10` | 93% | 5 s |
| `784 → 256 → 128 → 10` | 95% | 12 s |
| `784 → 512 → 256 → 128 → 10` | 96% | 30 s |

The benchmarks were measured on an Intel Core i5-12500H with 16 GB of RAM, using a single thread and an MSVC Release build. Results may vary with the training configuration and hardware.

## Project Documents

- [Project assignment (French)](./TEI-S7-NNv2.pdf)
- [Project report (French)](./docs/report/index.pdf)

## Prerequisites

- C compiler (GCC, Clang, or MSVC)
- CMake 3.12+
- libpng
- zlib
- Go 1.25+ (for the web interface only)
- dirent (for MSVC builds only)

```bash
# Ubuntu / Debian
sudo apt-get install build-essential cmake libpng-dev zlib1g-dev

# macOS
brew install cmake libpng zlib

# Windows with MSVC (vcpkg)
vcpkg install libpng:x64-windows zlib:x64-windows dirent:x64-windows
```

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

For an optimized build:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## Tests

```bash
cd build
ctest --output-on-failure
```

## Training

The repository includes preprocessed test data and a trained model in `examples/mnist-pngs/`. To retrain the model from scratch, you need the complete PNG dataset:

```bash
git clone https://github.com/rasbt/mnist-pngs mnist-pngs
```

Start training:

```bash
./build/examples/mnist_train mnist-pngs
```

Several training configurations are available in `examples/`:

| Example | Activation | Loss | Optimizer |
|---------|------------|------|-----------|
| `mnist_train.c` | Sigmoid | MSE | SGD |
| `mnist_train_sigmoid_sgd_mse.c` | Sigmoid | MSE | SGD |
| `mnist_train_relu_softmax_adam_cce.c` | ReLU / Softmax | Cross-Entropy | Adam |
| `mnist_train_tanh_softmax_sgd_cce.c` | Tanh / Softmax | Cross-Entropy | SGD |
| `mnist_train_architectures.c` | Sigmoid | MSE | SGD (compares several architectures) |

## Web Interface

The Go application uses Gin to recognize hand-drawn digits in real time. The server loads the binary model at startup and performs forward propagation in pure Go without cgo.

The interface preprocesses each drawing in the browser by centering it and scaling it to 28x28 pixels before sending it to the server.

## Using the Library

```c
#include "neuralnet.h"

const size_t layers[] = {784, 256, 128, 10};
Network *net = nn_network_create(layers, 4);

// Configure training
nn_network_set_training_config(net,
                               ACTIVATION_RELU,
                               ACTIVATION_SOFTMAX,
                               LOSS_CATEGORICAL_CROSS_ENTROPY,
                               OPTIMIZER_ADAM);

nn_network_train(net, train_data, 0.001f, 10);
float acc = nn_network_evaluate(net, test_data);
nn_network_save(net, "model.bin");
nn_network_free(net);
```

Available activations: `SIGMOID`, `RELU`, `SOFTMAX`, `TANH`, `LEAKY_RELU`, `LINEAR`

Loss functions: `MSE`, `MAE`, `BINARY_CROSS_ENTROPY`, `CATEGORICAL_CROSS_ENTROPY`

Optimizers: `SGD`, `ADAM`

## Project Structure

```
nn/
├── docs/
│   ├── report/                # project report and LaTeX source
│   └── image.png              # web interface screenshot
├── include/
│   ├── neuralnet.h            # main header
│   └── nn/
│       ├── matrix.h           # matrices and vectors
│       ├── mnist.h            # MNIST loading
│       ├── network.h          # neural network
│       └── nn_functions.h     # activations, losses, and optimizers
├── src/                       # implementation
├── examples/                  # training examples and bundled model
├── tests/                     # unit and integration tests
├── web/                       # Go web interface
│   ├── main.go                # Gin server
│   ├── nn/network.go          # inference in Go
│   ├── templates/index.html  # canvas interface
│   └── static/app.js          # client-side logic
├── CMakeLists.txt
├── CMakePresets.json
└── TEI-S7-NNv2.pdf            # project assignment
```

## Technical Notes

- The MNIST loader uses a binary cache (`train.bin`/`test.bin`) to speed up subsequent runs (a few seconds instead of several minutes for 70K PNG files).
- Training runs without dynamic memory allocation during forward and backward passes by using preallocated buffers.
- The web interface centers drawn digits by their center of mass during preprocessing, reproducing the original MNIST preprocessing method.
