/**
 * @file network.h
 * @brief Neural network structure and high-level operations
 *
 * This module defines the complete neural network as a sequence of layers
 * and provides high-level operations for inference and training.
 *
 * REQUIRED FUNCTIONALITY (from project PDF section 5):
 *
 * 1. Network creation with variable number of layers and sizes
 *    - Input: array of layer sizes, e.g., [784, 128, 64, 10]
 *    - Creates layers with appropriate dimensions
 *
 * 2. Network inference (forward propagation)
 *    - Input: image vector (784 elements for MNIST)
 *    - Output: class probabilities (10 elements)
 *    - Applies each layer sequentially
 *
 * 3. Network training (backpropagation + SGD)
 *    - Updates weights and biases based on loss gradient
 *    - See optimizer.h for SGD implementation
 *
 * 4. Network destruction (free all memory)
 *
 * NETWORK STRUCTURE:
 * - Array of Layer pointers
 * - Number of layers
 * - Layer sizes array (for reference)
 *
 * INFERENCE FLOW (from PDF equation 5):
 *   y_hat = argmax(sigma(W_2^T * sigma(W_1^T * sigma(W_0^T * x + B_0) + B_1) + B_2))
 *
 * For MNIST:
 * - Input layer: 784 neurons (28x28 pixels)
 * - Hidden layers: your choice (e.g., 128, 64)
 * - Output layer: 10 neurons (digits 0-9)
 */

#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include <stddef.h>
#include "layer.h"

/* TODO: Define Network structure */

/* TODO: Network creation from layer sizes array */
/* Example: nn_network_create([784, 128, 10], 3) creates 2-layer network */

/* TODO: Network destruction */

/* TODO: Forward propagation (inference) */
/* Returns output vector, class = argmax(output) */

/* TODO: Predict function - returns predicted class (0-9) */

/* TODO: Save/load network weights (optional but useful) */

#endif /* NN_NETWORK_H */
