/**
 * @file matrix.c
 * @brief Matrix operations implementation
 *
 * From PDF eq. (3): x^(k+1) = sigma(W_k^T * x^(k) + B_k)
 * Needs: create/destroy, matrix-vector multiply (W^T * x), vector addition
 */

#include "nn/matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* TODO: Implement matrix functions */
