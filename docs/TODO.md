# Implementation Checklist

Maps to the 5 requirements from TEI-S7-NN.pdf (Section 5).

## PDF Requirements

- [ ] **Req 1**: Gradient computation (dl/dW, dl/dB) — backpropagation in `network.c`
- [ ] **Req 2**: Create network with variable layers/sizes — `network_create()` in `network.c`
- [ ] **Req 3**: Inference function — `network_forward()` / `network_predict()` in `network.c`
- [ ] **Req 4**: Quality measurement — `network_accuracy()` in `network.c`
- [ ] **Req 5**: Training function (SGD) — `network_train()` in `network.c`

---

## Module Checklist

### matrix.c
- [ ] `matrix_create(rows, cols)` / `matrix_destroy(m)`
- [ ] `matrix_vector_multiply(W, x, result)` — W^T * x
- [ ] `vector_add(a, b, result)` — element-wise addition

### nn_functions.c
- [ ] `sigmoid(x)` / `sigmoid_derivative(output)`
- [ ] `relu(x)` / `relu_derivative(x)`
- [ ] `one_hot(label, num_classes)` — delta(y)
- [ ] `mse_loss(prediction, label)` — ||pred - delta(y)||^2
- [ ] `mse_gradient(prediction, label)` — 2*(pred - delta(y))

### network.c
- [ ] Network structure (layers with W, B, stored activations)
- [ ] `network_create(sizes[], num_sizes)` / `network_destroy(net)`
- [ ] `network_forward(net, input)` — full forward pass
- [ ] `network_predict(net, input)` — argmax of output
- [ ] Backpropagation — compute dl/dW and dl/dB for each layer
- [ ] `network_train(net, dataset, learning_rate, epochs)` — SGD loop
- [ ] `network_accuracy(net, dataset)` — correct / total

### mnist.c
- [ ] `mnist_load_png(filepath)` — single PNG to float[784]
- [ ] `mnist_load_dataset(base_path, "train"/"test")` — load all images
- [ ] `mnist_free_dataset(dataset)`
- [ ] `mnist_shuffle(dataset)` — Fisher-Yates shuffle

---

## Verification

- Network: [784, 128, 10]
- Learning rate: 0.01
- Epochs: 10
- Expected: >95% test accuracy
