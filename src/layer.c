/**
 * @file layer.c
 * @brief Layer operations implementation
 *
 * Implement the functions declared in nn/layer.h here.
 *
 * IMPLEMENTATION NOTES:
 *
 * Layer creation:
 * 1. Allocate Layer structure
 * 2. Create weight matrix W (input_size x output_size)
 * 3. Create bias vector B (output_size)
 * 4. Initialize weights randomly
 * 5. Initialize biases to zero (common practice)
 *
 * Random initialization example (Xavier):
 *   weight = ((double)rand() / RAND_MAX - 0.5) * 2.0 / sqrt(input_size)
 *
 * Forward pass:
 * 1. z = W^T * x + B  (pre-activation)
 * 2. a = activation(z) (output)
 * 3. Store z and a for backpropagation
 *
 * Don't forget to seed random: srand(time(NULL)) in main or network creation
 */

#include "nn/layer.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* TODO: Implement layer functions */
