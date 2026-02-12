# Task Distribution

## Mikita

### 1. Implement `nn_one_hot()`
**File:** `src/nn_functions.c`

Create the one-hot encoding function as defined in PDF equation 7. This function converts an integer label (0-9) into a vector where only the position corresponding to the label is 1, and all others are 0.

### 2. Implement `nn_network_evaluate()`
**File:** `src/network.c`

Create a function that calculates the accuracy of the network on a given MNIST dataset. It should run inference on all samples and return the percentage of correct predictions.

### 3. Implement `nn_network_train()`
**File:** `src/network.c`

Create the main training loop that iterates over the dataset for multiple epochs. It should shuffle data between epochs and call `nn_network_train_sample()` for each example. Print progress after each epoch.

### 4. Complete `mnist_train.c`
**File:** `examples/mnist_train.c`

Wire together all components into a complete training pipeline:
- Load training and test datasets
- Create the neural network
- Train the network
- Evaluate and print final accuracy

---

## Daris

### 1. Complete `nn_network_backward()`
**File:** `src/network.c`

Fix and complete the backpropagation function. It should correctly compute gradients dL/dW for each layer using the chain rule as described in PDF equations 11-12.

### 2. Implement `nn_network_train_sample()`
**File:** `src/network.c`

Create a function that trains the network on a single input-target pair. It should perform forward pass, backward pass, and update weights using the learning rate.

---

## File Ownership (to avoid conflicts)

| File | Mikita | Daris |
|------|--------|-------|
| `nn_functions.c` | nn_one_hot | - |
| `network.c` | nn_network_evaluate, nn_network_train | nn_network_backward, nn_network_train_sample |
| `mnist_train.c` | Full ownership | - |

## Dependency Order

```
nn_one_hot() ─────────────────────────────────┐
                                              ▼
nn_network_backward() ──► nn_network_train_sample() ──► nn_network_train() ──► mnist_train.c
        (Daris)                  (Daris)                    (Mikita)              (Mikita)
```

Daris completes backward and train_sample first, then Mikita can implement train and integrate everything.
