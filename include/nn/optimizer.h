/**
 * @file optimizer.h
 * @brief Training optimization algorithms
 *
 * This module implements optimization algorithms for updating network
 * weights during training. The primary algorithm is Stochastic Gradient
 * Descent (SGD) as specified in the project PDF.
 *
 * REQUIRED FUNCTIONALITY:
 *
 * 1. SGD update rule (from PDF Algorithm 1):
 *    W_k = W_k - eta * (dl/dW_k)
 *    B_k = B_k - eta * (dl/dB_k)
 *
 *    Where eta is the learning rate.
 *
 * 2. Backpropagation to compute gradients:
 *    This is the KEY part of the project (PDF section 5, point 1)
 *
 *    For each layer k (from output to input):
 *    - Compute dl/dz^(k) (gradient w.r.t. pre-activation)
 *    - Compute dl/dW_k = x^(k) * (dl/dz^(k))^T
 *    - Compute dl/dB_k = dl/dz^(k)
 *    - Propagate: dl/dx^(k) = W_k * dl/dz^(k)
 *
 * BACKPROPAGATION DERIVATION HINTS:
 * Chain rule: dl/dW_k = dl/dx^(k+1) * dx^(k+1)/dz^(k) * dz^(k)/dW_k
 *
 * Where:
 * - z^(k) = W_k^T * x^(k) + B_k (pre-activation)
 * - x^(k+1) = sigma(z^(k)) (post-activation)
 * - dz^(k)/dW_k involves x^(k)
 * - dx^(k+1)/dz^(k) = sigma'(z^(k))
 *
 * OPTIONAL OPTIMIZERS:
 * - SGD with momentum
 * - Adam
 *
 * TRAINING LOOP STRUCTURE:
 * for each epoch:
 *     for each sample (x, y):
 *         1. Forward pass (store intermediate values)
 *         2. Compute loss
 *         3. Backward pass (compute gradients)
 *         4. Update weights with SGD
 */

#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

#include <stddef.h>
#include "network.h"

/* TODO: Define optimizer configuration (learning rate, etc.) */

/* TODO: Backpropagation function */
/* Computes gradients for all layers given:
 * - Network with stored forward pass values
 * - Loss gradient (dl/dx for output layer)
 */

/* TODO: SGD update function */
/* Updates all weights and biases using computed gradients */

/* TODO: Training function for single sample */
/* Combines forward, loss, backward, and update */

/* TODO: Training function for entire dataset (epoch) */
/* Iterates over all samples, optionally shuffles */

#endif /* NN_OPTIMIZER_H */
