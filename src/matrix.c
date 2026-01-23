/**
 * @file matrix.c
 * @brief Matrix operations implementation
 *
 * Implement the functions declared in nn/matrix.h here.
 *
 * IMPLEMENTATION PRIORITY:
 * 1. Matrix structure and create/destroy functions
 * 2. Matrix-vector multiplication (most important for forward pass)
 * 3. Element-wise addition (for adding bias)
 * 4. Other utilities as needed
 *
 * TIPS:
 * - Use row-major order for matrix storage
 * - Access element (i,j) as: data[i * cols + j]
 * - For W^T * x, you can either transpose W or adjust multiplication
 */

#include "nn/matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* TODO: Implement matrix functions */
