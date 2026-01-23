/**
 * @file loss.c
 * @brief Loss functions implementation
 *
 * Implement the functions declared in nn/loss.h here.
 *
 * IMPLEMENTATION NOTES:
 *
 * One-hot encoding:
 *   Input: label = 3, num_classes = 10
 *   Output: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
 *
 *   for i in 0..num_classes:
 *       onehot[i] = (i == label) ? 1.0 : 0.0
 *
 * MSE Loss (from PDF equation 9):
 *   l(x, y) = ||x - delta(y)||^2 = sum((x[i] - onehot[i])^2)
 *
 *   Example:
 *     prediction = [0.1, 0.1, 0.1, 0.7, 0.0, ...]
 *     label = 3
 *     onehot = [0, 0, 0, 1, 0, ...]
 *     loss = (0.1-0)^2 + (0.1-0)^2 + (0.1-0)^2 + (0.7-1)^2 + ...
 *
 * MSE Gradient:
 *   dl/dx[i] = 2 * (x[i] - onehot[i])
 *
 *   This gradient is the starting point for backpropagation.
 */

#include "nn/loss.h"
#include <stdlib.h>
#include <math.h>

/* TODO: Implement loss functions */
