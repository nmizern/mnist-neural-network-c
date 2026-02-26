# Task Distribution

## Completed

### Mikita
- [x] `nn_one_hot()` - One-hot encoding function (nn_functions.c)
- [x] `nn_network_evaluate()` - Evaluate accuracy on mnist dataset (network.c)
- [x] `nn_network_train()` - Main training loop (network.c)
- [x] `mnist_train.c` - Full training pipeline (examples/mnist_train.c)

### Daris
- [x] `nn_network_backward()` - Backpropagation with weight updates (network.c)
- [x] `nn_network_save()` - Save network to file (network.c)
- [x] `nn_network_load()` - Load network from file (network.c)
- [x] `nn_network_copy()` - Copy network weights (network.c)

---

## Remaining Tasks — Performance Optimization

Training currently takes ~18 minutes due to 5,400,000 malloc/free calls (15 per sample x 180,000 samples). Goal: pre-allocate all buffers once and reuse them.

### Mikita — Struct changes + forward + binary cache (DO FIRST)

#### 1. ✅ Add `bias_input` to Layer struct (`network.h`)
#### 2. ✅ Add workspace buffers to Network struct (`network.h`)
#### 3. ✅ Allocate new buffers in `nn_network_create` (`network.c`)
#### 4. ✅ Free new buffers in `nn_network_free` (`network.c`)
#### 5. ✅ Rewrite `nn_network_forward` — zero malloc (`network.c`)

#### 6. Add binary cache for MNIST loading (`mnist.c` + `mnist.h`)
Loading 70,000 PNG files takes ~8 minutes due to filesystem overhead on Windows.
Save loaded dataset as a single `.bin` file, load from it on next run (~1 second).

- `mnist_save_binary(dataset, path)` — save images + labels to one binary file
- `mnist_load_binary(path)` — load dataset from binary file
- Update `mnist_train.c` to try `.bin` first, fallback to PNG, then save `.bin`

---

### Daris — Backward + train + evaluate (AFTER MIKITA PUSHES)

#### 1. Rewrite `nn_network_backward` — zero malloc (`network.c`)
Use workspace buffers instead of create_vector/free_vector.
Eliminates ~8 malloc/free per sample = 1,440,000 total.

#### 2. Update `nn_network_train` — pre-allocate input/target (`network.c`)
Create input (784) and target (1) once before loop, reuse with memcpy.
Eliminates 2 malloc/free per sample = 360,000 total.

#### 3. Update `nn_network_evaluate` — pre-allocate input (`network.c`)
Create input (784) once before loop, reuse with memcpy.

#### 4. Build and test
Verify same ~92% accuracy, measure speedup.

---

## Workflow to Avoid Conflicts

```
1. Mikita: struct changes + forward + binary cache → push
2. Daris:  git pull → backward + train + evaluate + test → push
```

## File Ownership

| File | Mikita | Daris |
|------|--------|-------|
| `network.h` | Layer/Network struct changes | - |
| `network.c` | create, free, forward | backward, train, evaluate |
| `mnist.h/c` | binary cache | - |
| `mnist_train.c` | binary cache integration | - |
