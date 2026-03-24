#ifndef NN_MATRIX_H
#define NN_MATRIX_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

struct Matrix {
    size_t rows;
    size_t cols;
    float *data;
};
typedef struct Matrix Matrix;

struct Vector {
    size_t size;
    float *data;
};
typedef struct Vector Vector;

/* Création et destruction */
Matrix* create_matrix(size_t rows, size_t cols);
Matrix* create_identity_matrix(size_t rows, size_t cols);
Matrix* create_rndm_matrix(size_t rows, size_t cols);
void free_matrix(Matrix* mat);

Vector* create_vector(size_t size);
Vector* create_rndm_vector(size_t size);
void free_vector(Vector* vec);

/* Multiplication */
void mat_vec_mul(const Matrix* mat, Vector* vec, Vector* result);
void mat_mat_mul(const Matrix* A, const Matrix* B, Matrix* result);

/* Opérations élément par élément */
float dot_product(const Vector* v1, const Vector* v2);
Vector* add_vectors(const Vector* v1, const Vector* v2);
Vector* multiply_vectors(const Vector* v1, const Vector* v2);
Vector* divide_vectors(const Vector* v1, const Vector* v2);
Vector* subtract_vectors(const Vector* v1, const Vector* v2);
Matrix* transpose_matrix(const Matrix* mat);

/* Utilitaires */
void print_matrix(const Matrix* mat);
void print_matrix_dimensions(const Matrix* mat);
void print_vector(const Vector* vec);
void copy_matrix(const Matrix* src, Matrix* dest);
void copy_vector(const Vector* src, Vector* dest);

#endif // NN_MATRIX_H
