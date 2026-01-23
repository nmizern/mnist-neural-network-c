/**
 * @file loss.h
 * @brief Loss functions for training
 *
 * This module implements loss functions that measure the difference
 * between network predictions and true labels.
 *
 * REQUIRED FUNCTIONALITY:
 *
 * 1. One-hot encoding (from PDF equations 6-7):
 *    delta(y)[i] = 1 if i == y, else 0
 *    Example: delta(3) = [0,0,0,1,0,0,0,0,0,0]
 *
 * 2. Mean Squared Error loss (from PDF equation 9):
 *    l(x, y) = ||x - delta(y)||^2
 *    Where x is network output, y is true label
 *
 * 3. Loss derivative for backpropagation:
 *    dl/dx = 2 * (x - delta(y))
 *
 * OPTIONAL (for better classification):
 * - Cross-entropy loss: -sum(delta(y) * log(x))
 *   Works better with softmax output
 *
 * MATHEMATICAL CONTEXT:
 * During training, we minimize the expected loss over the dataset:
 *   E[l(x, y)] over all (x, y) in training set
 *
 * The gradient dl/dx is propagated backward through the network
 * to compute dl/dW and dl/dB for each layer.
 */

#ifndef NN_LOSS_H
#define NN_LOSS_H

#include <stddef.h>

/* TODO: One-hot encoding function */
/* Input: label (0-9), output: vector of size num_classes */

/* TODO: MSE loss computation */
/* Input: prediction vector, true label */
/* Output: scalar loss value */

/* TODO: MSE loss gradient */
/* Input: prediction vector, true label */
/* Output: gradient vector (same size as prediction) */

/* OPTIONAL: Cross-entropy loss and gradient */

#endif /* NN_LOSS_H */
