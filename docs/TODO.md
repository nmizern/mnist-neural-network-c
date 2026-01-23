# Implementation Checklist

This checklist maps to the project requirements from TEI-S7-NN.pdf.

## Project Requirements (Section 5)

- [ ] **1. Understand gradient calculation** (dℓ/dW and dℓ/dB)
  - Study backpropagation algorithm
  - Implement in `optimizer.c`

- [ ] **2. Create network with variable layers/sizes**
  - Implement in `network.c`: `network_create(sizes[], num_sizes)`

- [ ] **3. Inference function**
  - Implement in `network.c`: `network_forward(net, input)`

- [ ] **4. Quality measurement function**
  - Implement in `metrics.c`: `evaluate_accuracy(net, dataset)`

- [ ] **5. Training function**
  - Implement in `optimizer.c`: `train_epoch(net, dataset, lr)`

---

## Module Implementation Checklist

### matrix.c
- [ ] Matrix structure definition
- [ ] `matrix_create(rows, cols)`
- [ ] `matrix_destroy(m)`
- [ ] `matrix_vector_multiply(W, x, result)` - W^T * x
- [ ] `vector_add(a, b, result)` - element-wise addition

### activation.c
- [ ] `sigmoid(x)` - 1 / (1 + exp(-x))
- [ ] `sigmoid_derivative(output)` - output * (1 - output)
- [ ] `relu(x)` - max(0, x)
- [ ] `relu_derivative(x)` - x > 0 ? 1 : 0
- [ ] Apply activation to vector

### layer.c
- [ ] Layer structure (weights, biases, activation type)
- [ ] `layer_create(input_size, output_size, activation)`
- [ ] `layer_destroy(layer)`
- [ ] `layer_forward(layer, input)` - compute z and apply activation
- [ ] Weight initialization (Xavier or simple random)

### network.c
- [ ] Network structure (array of layers)
- [ ] `network_create(sizes[], num_sizes)`
- [ ] `network_destroy(net)`
- [ ] `network_forward(net, input)` - full forward pass
- [ ] `network_predict(net, input)` - returns argmax class

### loss.c
- [ ] `one_hot(label, num_classes)` - create one-hot vector
- [ ] `mse_loss(prediction, label)` - ||pred - onehot(label)||^2
- [ ] `mse_gradient(prediction, label)` - 2 * (pred - onehot)

### optimizer.c (MOST IMPORTANT)
- [ ] Store forward pass intermediate values (z, a for each layer)
- [ ] `backpropagate(net, loss_gradient)` - compute all gradients
- [ ] `sgd_update(net, learning_rate)` - apply gradients
- [ ] `train_sample(net, input, label, lr)` - single sample training
- [ ] `train_epoch(net, dataset, lr)` - train on full dataset

### mnist.c
- [ ] Dataset structure (images, labels, count)
- [ ] `mnist_load_image(path)` - load single PNG as float array
- [ ] `mnist_load_dataset(base_path, split)` - load train or test set
- [ ] `mnist_free(dataset)`

### metrics.c
- [ ] `calculate_accuracy(net, dataset)` - correct / total
- [ ] Print evaluation results

---

## Testing Checklist

- [ ] test_matrix.c - Matrix operations work correctly
- [ ] test_activation.c - Activation functions and derivatives
- [ ] test_network.c - Network creation and forward pass

---

## Examples Checklist

- [ ] xor_example.c - XOR problem trains successfully
- [ ] mnist_train.c - MNIST achieves >90% accuracy

---

## Verification

### XOR Test
After implementing all modules, XOR should work:
- Network: [2, 4, 1]
- Learning rate: 0.5
- Epochs: 10000
- Expected: All 4 XOR cases correct

### MNIST Test
- Network: [784, 128, 10]
- Learning rate: 0.01
- Epochs: 10
- Expected: >95% test accuracy
