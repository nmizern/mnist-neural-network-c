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

### Mikita — Struct changes + backward + train (DO FIRST)

#### 1. Add `bias_input` to Layer struct (`network.h`)
Add `Vector *bias_input;` to Layer struct.
Each layer needs an input+bias vector for matrix multiplication. Currently allocated/freed every call.

#### 2. Add workspace buffers to Network struct (`network.h`)
Add to Network struct:
- `Vector *work_predicted` — size = output layer
- `Vector *work_one_hot` — size = output layer
- `Vector *work_delta` — size = max(layer_sizes[1..n])
- `Vector *work_prev_delta` — size = max(layer_sizes[1..n])
- `Vector *work_act_deriv` — size = max(layer_sizes[1..n])

These replace all temporary vectors created inside backward.

#### 3. Allocate new buffers in `nn_network_create` (`network.c`)
- `bias_input` per layer: size = input_size + 1
- Find max_size = max of layer_sizes[1..n]
- Allocate 5 workspace vectors

#### 4. Free new buffers in `nn_network_free` (`network.c`)
Free `bias_input` per layer + 5 workspace vectors.

#### 5. Rewrite `nn_network_backward` — zero malloc (`network.c`)
Use workspace buffers instead of create_vector/free_vector.
Eliminates ~8 malloc/free per sample = 1,440,000 total.

#### 6. Update `nn_network_train` — pre-allocate input/target (`network.c`)
Create input (784) and target (1) once before loop, reuse with memcpy.
Eliminates 2 malloc/free per sample = 360,000 total.

---

### Daris — Forward + evaluate (AFTER MIKITA PUSHES)

#### 1. Rewrite `nn_network_forward` — zero malloc (`network.c`)
Use `layer->input_cache` as current, `layer->bias_input` for input+bias, `layer->output_cache` as result.
Eliminates ~5 malloc/free per call = 900,000 total.

#### 2. Update `nn_network_evaluate` — pre-allocate input (`network.c`)
Create input (784) once before loop, reuse with memcpy.

#### 3. Build and test
Verify same ~92% accuracy, measure speedup.

---

## Workflow to Avoid Conflicts

```
1. Mikita: struct changes + backward + train → push
2. Daris:  git pull → forward + evaluate + test → push
```

## File Ownership

| File | Mikita | Daris |
|------|--------|-------|
| `network.h` | Layer/Network struct changes | - |
| `network.c` | backward, train, create, free | forward, evaluate |
| `mnist_train.c` | done | - |
