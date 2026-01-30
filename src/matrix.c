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

Matrix* create_rndm_matrix(size_t rows, size_t cols) {
    Matrix* mat = create_matrix(rows, cols);
    for (size_t i = 0; i < rows * cols; i++) {
        mat->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Valeurs aléatoires dans [-1, 1]
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
        vec->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Valeurs aléatoires dans [-1, 1]
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