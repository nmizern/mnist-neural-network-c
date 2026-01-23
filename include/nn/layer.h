/**
 * @file layer.h
 * @brief Neural network layer structure and operations
 *
 * This module defines a single fully-connected (dense) layer.
 * Each layer contains weights, biases, and knows its activation function.
 *
 * REQUIRED FUNCTIONALITY:
 * - Layer structure containing:
 *   - Weights matrix W (input_size x output_size)
 *   - Bias vector B (output_size)
 *   - Activation function type
 *   - Input/output sizes
 *
 * - Layer creation with random weight initialization
 * - Layer destruction (free memory)
 * - Layer forward pass: output = activation(W^T * input + B)
 *
 * FOR BACKPROPAGATION (training):
 * - Store intermediate values during forward pass:
 *   - z = W^T * x + B (pre-activation)
 *   - a = sigma(z) (post-activation, layer output)
 * - These are needed to compute gradients
 *
 * WEIGHT INITIALIZATION:
 * - Random initialization is crucial for breaking symmetry
 * - Consider Xavier/He initialization for better convergence:
 *   - Xavier: weights ~ N(0, 1/n_in)
 *   - He: weights ~ N(0, 2/n_in) for ReLU
 * - Simple approach: small random values like [-0.5, 0.5] / sqrt(n_in)
 *
 * MATHEMATICAL CONTEXT (from PDF equation 3):
 *   x^(k+1) = sigma(W_k^T * x^(k) + B_k)
 */

#ifndef NN_LAYER_H
#define NN_LAYER_H

#include <stddef.h>
#include "matrix.h"
#include "activation.h"

/* TODO: Define Layer structure */

/* TODO: Layer creation with specified sizes and activation */

/* TODO: Layer destruction */

/* TODO: Layer forward pass */

/* TODO: Storage for intermediate values (for backprop) */

#endif /* NN_LAYER_H */
