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

### Matrix and Vectors
- [x] Matrix structure and Vectors definition
- [x] `matrix_create(rows, cols)`
- [x] `matrix_destroy(m)`
- [x] `matrix_vector_multiply(W, x, result)` - W^T * x
- [x] `vector_add(a, b, result)` - element-wise addition

### Activation
- [ ] `sigmoid(x)` - 1 / (1 + exp(-x))
- [ ] `sigmoid_derivative(output)` - output * (1 - output)
- [ ] `relu(x)` - max(0, x)
- [ ] `relu_derivative(x)` - x > 0 ? 1 : 0
- [ ] Apply activation to vector

### Layers
- [ ] Layer structure (weights, biases, activation type)
- [ ] `layer_create(input_size, output_size, activation)`
- [ ] `layer_destroy(layer)`
- [ ] `layer_forward(layer, input)` - compute z and apply activation
- [ ] Weight initialization (Xavier or simple random)

### Network Structure
- [X] Network structure (array of layers)
- [X] `network_create(sizes[], num_sizes)`
- [X] `network_destroy(net)`
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
---

## Testing Checklist

- [x] test_matrix.c - Matrix operations work correctly
- [ ] test_activation.c - Activation functions and derivatives
- [ ] test_network.c - Network creation and forward pass

---

## Examples Checklist

- [ ] xor_example.c - XOR problem trains successfully
- [ ] mnist_train.c - MNIST achieves >90% accuracy
