/**
 * @file matrix.h
 * @brief Matrix operations for neural network computations
 *
 * This module provides fundamental matrix/vector operations needed for
 * neural network forward and backward propagation.
 *
 * REQUIRED FUNCTIONALITY:
 * - Matrix structure definition (rows, cols, data pointer)
 * - Matrix creation and destruction (memory management)
 * - Matrix-vector multiplication: y = W^T * x
 * - Matrix-matrix multiplication (for batch processing, optional)
 * - Element-wise operations (addition, subtraction)
 * - Vector dot product
 * - Matrix transpose (or handle transposed access)
 *
 * MATHEMATICAL CONTEXT:
 * In the project PDF, the layer equation is:
 *   x^(k+1) = sigma(W_k^T * x^(k) + B_k)
 *
 * So you need efficient matrix-vector multiplication where W is (N_k x N_{k+1})
 * and x is (N_k x 1), resulting in (N_{k+1} x 1) output.
 *
 * MEMORY MANAGEMENT:
 * - All matrices should be allocated on heap
 * - Provide clear create/destroy functions
 * - Consider using contiguous memory for cache efficiency
 */

#ifndef NN_MATRIX_H
#define NN_MATRIX_H

#include <stddef.h>

// Cette structure de matrice sera utilisé pour definir les matrices
struct Matrice
{
    size_t rows;
    size_t cols;
    float *data;
};

/* TODO: Define Matrix structure */

/* TODO: Matrix creation and destruction functions */

/* TODO: Matrix-vector multiplication */

/* TODO: Element-wise operations */

/* TODO: Utility functions (print, copy, etc.) */

#endif /* NN_MATRIX_H */
