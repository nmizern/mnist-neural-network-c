/**
 * @file matrix.h
 * @brief Matrix structure and operations
 *
 * Foundation for the neural network. Used to store weights (W),
 * biases (B), and intermediate vectors (x) at each layer.
 *
 * From PDF equation (3): x^(k+1) = sigma(W_k^T * x^(k) + B_k)
 * You need: matrix-vector multiply, vector addition, element-wise ops.
 */

#ifndef NN_MATRIX_H
#define NN_MATRIX_H

#include <stddef.h>

struct Matrice
{
    size_t rows;
    size_t cols;
    float *data;
};

/* TODO: Matrix create/destroy */
/* TODO: Matrix-vector multiplication (W^T * x) */
/* TODO: Vector addition (z + B) */
/* TODO: Element-wise operations */

#endif /* NN_MATRIX_H */
