/**
 * @file activation.c
 * @brief Activation functions implementation
 *
 * Implement the functions declared in nn/activation.h here.
 *
 * IMPLEMENTATION NOTES:
 *
 * Sigmoid:
 *   sigma(x) = 1.0 / (1.0 + exp(-x))
 *   - Use <math.h> for exp()
 *   - Handle overflow: for very negative x, exp(-x) is huge
 *     Clamp x to reasonable range, e.g., [-500, 500]
 *
 * Sigmoid derivative:
 *   sigma'(x) = sigma(x) * (1 - sigma(x))
 *   If you have output = sigma(x), then: derivative = output * (1 - output)
 *
 * ReLU:
 *   relu(x) = x > 0 ? x : 0
 *
 * ReLU derivative:
 *   relu'(x) = x > 0 ? 1 : 0
 */

#include "nn/nn_functions.h"
#include <math.h>

/* TODO: Implement activation functions */
