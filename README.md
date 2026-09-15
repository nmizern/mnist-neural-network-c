[**English**](./README.md) | [Français](./README.fr.md)

# Neural Network in C

A C library for dense (fully connected) neural networks for MNIST classification, with a web interface written in Go.

https://github.com/nmizern/mnist-neural-network-c

**Authors:** Mikita Mizerkin, Idirene Daris

**Course:** TEI S7 - Neural Networks Project

> **Successfully presented to the examination committee — grade: 19/20, the best result in the L3 E3A class.**
>
> Developed over 10 four-hour sessions, supplemented by independent work at home.

![Web interface screenshot](./docs/image.png)

## Prerequisites

- C compiler (GCC, Clang, or MSVC)
- CMake 3.12+
- libpng

```bash
# Ubuntu / Debian
sudo apt-get install libpng-dev

# macOS
brew install libpng

# Windows (vcpkg)
vcpkg install libpng
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

The repository includes preprocessed data (`examples/mnist-pngs/test.bin`, `mnist_model.bin`). To retrain the model from scratch, you need the complete PNG dataset:

```bash
git clone https://github.com/rasbt/mnist-pngs examples/mnist-pngs
```

Start training:

```bash
./build/examples/mnist_train examples/mnist-pngs
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

A Go application using Gin for drawing and recognizing digits in real time.

```bash
cd web
go mod download
go run . -model ../examples/mnist-pngs/mnist_model.bin -addr :4343
```

Open `http://localhost:4343` in a browser.

The server loads the binary model at startup and performs forward propagation in pure Go (without cgo). The interface lets you draw a digit on a canvas. Preprocessing (centering and scaling to 28x28) is performed on the client before the image is sent to the server.

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
├── include/
│   ├── neuralnet.h            # main header
│   └── nn/
│       ├── matrix.h           # matrices and vectors
│       ├── mnist.h            # MNIST loading
│       ├── network.h          # neural network
│       └── nn_functions.h     # activations, losses, and optimizers
├── src/                       # implementation
├── examples/                  # training examples
├── tests/                     # unit tests
├── web/                       # Go web interface
│   ├── main.go                # Gin server
│   ├── nn/network.go          # inference in Go
│   ├── templates/index.html  # canvas interface
│   └── static/app.js          # client-side logic
├── CMakeLists.txt
└── CMakePresets.json
```

## Technical Notes

- The MNIST loader uses a binary cache (`train.bin`/`test.bin`) to speed up subsequent runs (a few seconds instead of several minutes for 70K PNG files).
- Training runs without dynamic memory allocation during forward and backward passes (using preallocated buffers).
- The web interface centers drawn digits by their center of mass during preprocessing, reproducing the original MNIST preprocessing method.

## License

University project - TEI S7.
