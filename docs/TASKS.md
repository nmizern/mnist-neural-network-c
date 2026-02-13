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

### Mikita - Implement `nn_network_train()` ⬅️ DO FIRST
**File:** `src/network.c` + `include/nn/network.h`

Main training loop over multiple epochs. For each epoch: shuffle dataset, iterate over all samples, call backward for each sample, optionally print progress.

**Signature:** `void nn_network_train(Network *network, mnist_dataset_t *train_data, float learning_rate, int epochs);`

**Note:** Since `nn_network_backward()` already does forward + gradient + weight update, you can call it directly for each sample (no need for separate train_sample).

### Daris - Complete `mnist_train.c` ⬅️ AFTER MIKITA PUSHES
**File:** `examples/mnist_train.c`

Update the file to perform actual training:
- Load train/test datasets (already done)
- Create network with chosen architecture (e.g., [784, 128, 10])
- Call `nn_network_train()` with learning rate and epochs
- Call `nn_network_evaluate()` on test set
- Print final accuracy
- Optionally save trained network

## Workflow to Avoid Conflicts

```
1. Mikita implements nn_network_train()
2. Mikita pushes to main
3. Daris does git pull
4. Daris implements mnist_train.c
5. Daris pushes to main
```

---

## File Ownership

| File | Mikita | Daris |
|------|--------|-------|
| `nn_functions.h/c` | ✅ nn_one_hot | - |
| `network.h/c` | ✅ nn_network_evaluate, nn_network_train | ✅ nn_network_backward, save/load/copy (done) |
| `mnist_train.c` | - | Full ownership |

## Dependency Order

```
[DONE] nn_one_hot()
[DONE] nn_network_backward()
[DONE] nn_network_evaluate()
         │
         ▼
nn_network_train() ──► mnist_train.c
     (Mikita)            (Daris)
```
