#ifndef NN_MATRIX_H
#define NN_MATRIX_H

#include <stddef.h>

typedef struct {
    size_t rows;
    size_t cols;
    float *data; 
} matrix_t;

matrix_t *matrix_create(size_t rows, size_t cols);

void matrix_destroy(matrix_t *m);

void matrix_vec_mul(const matrix_t *W, const float *x, float *result);

void vector_add(const float *a, const float *b, float *result, size_t n);

void matrix_random_init(matrix_t *m, float limit);

#endif 
