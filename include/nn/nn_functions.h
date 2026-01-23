/**
 * @file activation.h
 * @brief Activation functions and their derivatives
 *
 * This module implements non-linear activation functions used in neural networks.
 * Each activation function needs both forward (for inference) and derivative
 * (for backpropagation) implementations.
 *
 * REQUIRED FUNCTIONALITY:
 * - Sigmoid: sigma(x) = 1 / (1 + e^(-x))
 *   Derivative: sigma'(x) = sigma(x) * (1 - sigma(x))
 *
 * - ReLU: relu(x) = max(0, x)
 *   Derivative: relu'(x) = 1 if x > 0, else 0
 *
 * OPTIONAL (for better performance):
 * - Softmax (for output layer in classification)
 * - Tanh
 *
 * IMPLEMENTATION NOTES:
 * - Functions should work element-wise on vectors
 * - For backpropagation, you need derivatives
 * - Consider numerical stability (e.g., sigmoid overflow for large negative x)
 *
 * MATHEMATICAL CONTEXT (from PDF):
 * The activation function sigma is applied component-wise after the linear
 * transformation: x^(k+1)[j] = sigma(w_{k,j}^T * x^(k) + b_{k,j})
 */

#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include <stddef.h>

/* TODO: Define activation function type enum (SIGMOID, RELU, etc.) */
void nn_activation_forward(/* parameters */);
void nn_activation_derivative(/* parameters */);

/* TODO: Forward activation functions */
void nn_activation_forward(/* parameters */);
void nn_activation_derivative(/* parameters */);
/* Apply activation to entire vector in-place or to output */

/* TODO: Derivative functions for backpropagation */
/* Note: sigmoid derivative can be computed from output: sigma'(x) = sigma(x)*(1-sigma(x)) */
/* ReLU derivative needs original input to check if x > 0 */

#endif /* NN_ACTIVATION_H */
