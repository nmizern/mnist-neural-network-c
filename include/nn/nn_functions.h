/**
 * @file nn_functions.h
 * @brief Activation functions, loss function, and their derivatives
 *
 * Contains all mathematical functions needed by the network:
 *
 * ACTIVATION (PDF eq. 4):
 *   sigmoid(x) = 1 / (1 + exp(-x))
 *   ReLU(x)    = max(x, 0)
 *   + their derivatives for backpropagation
 *
 * LOSS (PDF eq. 9):
 *   l(x, y) = ||x - delta(y)||^2    (MSE)
 *   delta(y) = one-hot encoding      (PDF eq. 7)
 *   + loss gradient: dl/dx = 2*(x - delta(y))
 */

#ifndef NN_FUNCTIONS_H
#define NN_FUNCTIONS_H

#include <stddef.h>

/* TODO: Activation functions (forward + derivative) */
/* TODO: One-hot encoding */
/* TODO: MSE loss computation */
/* TODO: MSE loss gradient (starting point for backpropagation) */

#endif /* NN_FUNCTIONS_H */
