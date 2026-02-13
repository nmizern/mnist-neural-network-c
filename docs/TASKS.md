# Task Distribution

## What Already Exists

### nn_functions.c (COMPLETE)
- `nn_sigmoid`, `nn_relu`, `nn_softmax` + derivatives
- `nn_activation_apply`, `nn_activation_derivative`
- `nn_loss_mse`, `nn_loss_mse_gradient`
- `nn_sgd_create`, `nn_sgd_free`, `nn_sgd_update`
- `nn_argmax`, `nn_accuracy` (takes arrays, not dataset)

### network.c (PARTIALLY COMPLETE)
- `nn_network_create`, `nn_network_free`
- `nn_network_forward`, `nn_network_predict`
- `nn_network_zero_gradients`, `nn_network_print`
- `nn_network_backward` - EXISTS BUT INCOMPLETE (has TODO, logic not finished)

### Declared but NOT implemented
- `nn_network_save`, `nn_network_load`, `nn_network_copy` (optional, not required by PDF)

---

## Mikita

### 1. Implement `nn_one_hot()`
**File:** `src/nn_functions.c` + `include/nn/nn_functions.h`

Create the one-hot encoding function as defined in PDF equation 7. Converts integer label (0-9) into a vector where position `label` is 1.0, all others are 0.0.

**Signature:** `void nn_one_hot(size_t label, size_t num_classes, Vector *result);`

### 2. Implement `nn_network_evaluate()`
**File:** `src/network.c` + `include/nn/network.h`

Evaluate network accuracy on mnist_dataset_t. Loops through all samples, calls `nn_network_predict()`, compares with labels, returns accuracy (0.0 to 1.0).

Note: `nn_accuracy()` already exists but takes arrays. This function wraps it for mnist_dataset_t.

**Signature:** `float nn_network_evaluate(const Network *network, const mnist_dataset_t *dataset);`

### 3. Implement `nn_network_train()`
**File:** `src/network.c` + `include/nn/network.h`

Main training loop over multiple epochs. For each epoch: shuffle dataset, call `nn_network_train_sample()` for each sample, optionally print progress.

**Signature:** `void nn_network_train(Network *network, mnist_dataset_t *train_data, float learning_rate, int epochs);`

### 4. Complete `mnist_train.c`
**File:** `examples/mnist_train.c`

Wire together all components:
- Load train/test datasets
- Create network with chosen architecture (e.g., [784, 128, 10])
- Call `nn_network_train()`
- Call `nn_network_evaluate()` on test set
- Print final accuracy

---

## Daris

### 1. Complete `nn_network_backward()`
**File:** `src/network.c` (lines 75-117)

The function exists but is incomplete. Fix the backpropagation logic:
- Correctly compute dL/dW for each layer using chain rule (PDF eq. 11-12)
- Update weights: W = W - learning_rate * gradient
- Fix memory leaks (some vectors allocated but not freed)

### 2. Implement `nn_network_train_sample()`
**File:** `src/network.c` + `include/nn/network.h`

Train on a single (input, target) pair. Calls backward which updates weights.

**Signature:** `void nn_network_train_sample(Network *network, const Vector *input, const Vector *target, float learning_rate);`

---

## File Ownership

| File | Mikita | Daris |
|------|--------|-------|
| `nn_functions.h/c` | nn_one_hot | - |
| `network.h/c` | nn_network_evaluate, nn_network_train | nn_network_backward (fix), nn_network_train_sample |
| `mnist_train.c` | Full ownership | - |

## Dependency Order

```
nn_one_hot() ──────────────────────────────────────┐
                                                   ▼
nn_network_backward() ──► nn_network_train_sample() ──► nn_network_train() ──► mnist_train.c
       (Daris)                   (Daris)                    (Mikita)              (Mikita)
```

Mikita can start with `nn_one_hot()` immediately.
`nn_network_evaluate()` only needs `nn_network_predict()` which already works.
`nn_network_train()` and `mnist_train.c` wait for Daris to finish `train_sample()`.
