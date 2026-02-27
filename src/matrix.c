#include "nn/matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Matrix* create_matrix(size_t rows, size_t cols) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (float*)calloc(rows * cols, sizeof(float));
    return mat;
}

Matrix* create_identity_matrix(size_t rows, size_t cols) {
    Matrix* mat = create_matrix(rows, cols);
    size_t min_dim = rows < cols ? rows : cols;
    for (size_t i = 0; i < min_dim; i++) {
        mat->data[i * cols + i] = 1.0f;
    }
    return mat;
}

Matrix* create_rndm_matrix(size_t rows, size_t cols) {
    Matrix* mat = create_matrix(rows, cols);
    for (size_t i = 0; i < rows * cols; i++) {
        mat->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    return mat;
}

void free_matrix(Matrix* mat) {
    if (mat) {
        free(mat->data);
        free(mat);
    }
}

Vector* create_vector(size_t size) {
    Vector* vec = (Vector*)malloc(sizeof(Vector));
    vec->size = size;
    vec->data = (float*)calloc(size, sizeof(float));
    return vec;
}

Vector* create_rndm_vector(size_t size) {
    Vector* vec = create_vector(size);
    for (size_t i = 0; i < size; i++) {
        vec->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    return vec;
}

void mat_vec_mul(const Matrix* mat, Vector* vec, Vector* result) {
    if (mat->cols != vec->size || mat->rows != result->size) {
        fprintf(stderr, "Dimension mismatch in mat_vec_mul\n");
        return;
    }
    for (size_t i = 0; i < mat->rows; i++) {
        result->data[i] = 0.0f;
        for (size_t j = 0; j < mat->cols; j++) {
            result->data[i] += mat->data[i * mat->cols + j] * vec->data[j];
        }
    }
}

void mat_mat_mul(const Matrix* A, const Matrix* B, Matrix* result) {
    if (A->cols != B->rows || A->rows != result->rows || B->cols != result->cols) {
        fprintf(stderr, "Dimension mismatch in mat_mat_mul\n");
        return;
    }
    Matrix* temp = create_matrix(A->rows, B->cols);
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < B->cols; j++) {
            temp->data[i * B->cols + j] = 0.0f;
            for (size_t k = 0; k < A->cols; k++) {
                temp->data[i * B->cols + j] += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
        }
    }
    copy_matrix(temp, result);
    free_matrix(temp);
}

float dot_product(const Vector* v1, const Vector* v2) {
    if (v1->size != v2->size) {
        fprintf(stderr, "Dimension mismatch in dot_product\n");
        return 0.0f;
    }
    float result = 0.0f;
    for (size_t i = 0; i < v1->size; i++) {
        result += v1->data[i] * v2->data[i];
    }
    return result;
}

void free_vector(Vector* vec) {
    if (vec) {
        free(vec->data);
        free(vec);
    }
}

/* Opérations sur les vecteurs */
Vector* add_vectors(const Vector* v1, const Vector* v2) {
    if (v1->size != v2->size) {
        fprintf(stderr, "Dimension mismatch in add_vectors\n");
        return NULL;
    }
    Vector* result = create_vector(v1->size);
    for (size_t i = 0; i < v1->size; i++) {
        result->data[i] = v1->data[i] + v2->data[i];
    }
    return result;
}

Vector* subtract_vectors(const Vector* v1, const Vector* v2) {
    if (v1->size != v2->size) {
        fprintf(stderr, "Dimension mismatch in subtract_vectors\n");
        return NULL;
    }
    Vector* result = create_vector(v1->size);
    for (size_t i = 0; i < v1->size; i++) {
        result->data[i] = v1->data[i] - v2->data[i];
    }
    return result;
}

Vector* multiply_vectors(const Vector* v1, const Vector* v2) {
    if (v1->size != v2->size) {
        fprintf(stderr, "Dimension mismatch in multiply_vectors\n");
        return NULL;
    }
    Vector* result = create_vector(v1->size);
    for (size_t i = 0; i < v1->size; i++) {
        result->data[i] = v1->data[i] * v2->data[i];
    }
    return result;
}

Vector* divide_vectors(const Vector* v1, const Vector* v2) {
    if (v1->size != v2->size) {
        fprintf(stderr, "Dimension mismatch in divide_vectors\n");
        return NULL;
    }
    Vector* result = create_vector(v1->size);
    for (size_t i = 0; i < v1->size; i++) {
        if (v2->data[i] != 0.0f) {
            result->data[i] = v1->data[i] / v2->data[i];
        } else {
            result->data[i] = 0.0f;
        }
    }
    return result;
}

Matrix* transpose_matrix(const Matrix* mat) {
    Matrix* result = create_matrix(mat->cols, mat->rows);
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            result->data[j * mat->rows + i] = mat->data[i * mat->cols + j];
        }
    }
    return result;
}

/* Utilitaires */
void print_matrix(const Matrix* mat) {
    printf("Matrix [%zu x %zu]:\n", mat->rows, mat->cols);
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            printf("%.4f ", mat->data[i * mat->cols + j]);
        }
        printf("\n");
    }
}

void print_vector(const Vector* vec) {
    printf("Vector [%zu]: ", vec->size);
    for (size_t i = 0; i < vec->size; i++) {
        printf("%.4f ", vec->data[i]);
    }
    printf("\n");
}

void print_matrix_dimentions(const Matrix* mat) {
    printf("Matrix dimensions: %zu x %zu\n", mat->rows, mat->cols);
}

void copy_matrix(const Matrix* src, Matrix* dest) {
    if (src->rows != dest->rows || src->cols != dest->cols) {
        fprintf(stderr, "Dimension mismatch in copy_matrix\n");
        return;
    }
    memcpy(dest->data, src->data, src->rows * src->cols * sizeof(float));
}

void copy_vector(const Vector* src, Vector* dest) {
    if (src->size != dest->size) {
        fprintf(stderr, "Dimension mismatch in copy_vector\n");
        return;
    }
    memcpy(dest->data, src->data, src->size * sizeof(float));
}