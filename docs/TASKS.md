# Task Distribution

## Completed

### Mikita
- [x] `nn_one_hot()` - One-hot encoding function (nn_functions.c)
- [x] `nn_network_evaluate()` - Evaluate accuracy on mnist dataset (network.c)

### Daris
- [x] `nn_network_backward()` - Backpropagation with weight updates (network.c)
- [x] `nn_network_save()` - Save network to file (network.c)
- [x] `nn_network_load()` - Load network from file (network.c)
- [x] `nn_network_copy()` - Copy network weights (network.c)

---

## Remaining Tasks

## Mikita

### 1. Implement `nn_network_train()`
**File:** `src/network.c` + `include/nn/network.h`

Main training loop over multiple epochs. For each epoch: shuffle dataset, iterate over all samples, call backward for each sample, optionally print progress.

**Signature:** `void nn_network_train(Network *network, mnist_dataset_t *train_data, float learning_rate, int epochs);`

**Note:** Since `nn_network_backward()` already does forward + gradient + weight update, you can call it directly for each sample (no need for separate train_sample).

### 2. Complete `mnist_train.c`
**File:** `examples/mnist_train.c`

Update the file to perform actual training:
- Load train/test datasets (already done)
- Create network with chosen architecture (e.g., [784, 128, 10])
- Call `nn_network_train()` with learning rate and epochs
- Call `nn_network_evaluate()` on test set
- Print final accuracy
- Optionally save trained network

---

## Daris

### 1. (Optional) Implement `nn_network_train_sample()`
**File:** `src/network.c` + `include/nn/network.h`

Wrapper around backward for training on a single sample. This is optional since `nn_network_backward()` already performs a complete training step.

**Signature:** `void nn_network_train_sample(Network *network, const Vector *input, const Vector *target, float learning_rate);`

---

## File Ownership

| File | Mikita | Daris |
|------|--------|-------|
| `nn_functions.h/c` | ✅ nn_one_hot | - |
| `network.h/c` | ✅ nn_network_evaluate, nn_network_train | ✅ nn_network_backward, save/load/copy |
| `mnist_train.c` | Full ownership | - |

## Dependency Order

```
[DONE] nn_one_hot()
[DONE] nn_network_backward()
[DONE] nn_network_evaluate()
         │
         ▼
nn_network_train() ──► mnist_train.c
     (Mikita)            (Mikita)
```

Mikita can now implement `nn_network_train()` and complete `mnist_train.c`.
