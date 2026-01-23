#ifndef NN_MATRIX_H
#define NN_MATRIX_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


// Cette structure de matrice sera utilisé pour definir les matrices dans le reseau de neurones
// Ces matrices seront representerons les poids, les biais, et les couches ou 'layers' du reseau
struct Matrix
{
    size_t rows;
    size_t cols;
    float *data;
};
typedef struct Matrix Matrix;

struct Vector
{
    size_t size;
    float *data;
};
typedef struct Vector Vector;


// Matrice creation et destruction
Matrix* create_matrix(size_t rows, size_t cols);
Matrix* create_rndm_matrix(size_t rows, size_t cols);
void free_matrix(Matrix* mat);

Vector* create_vector(size_t size);
Vector* create_rndm_vector(size_t size);
void free_vector(Vector* vec);


// Matrice-vecteur multiplication
// On multiplie la matrice mat (de taille rows x cols) avec le vecteur vec (de taille cols x 1)
// Le resultat est stocke dans le vecteur result (de taille rows x 1)
void mat_vec_mul(const Matrix* mat, Vector* vec, Vector* result);
void mat_mat_mul(const Matrix* A, const Matrix* B, Matrix* result);

// Element-wise operations
float dot_product(const Vector* v1, const Vector* v2);
void add_vectors(const Vector* v1, const Vector* v2, Vector* result);
void subtract_vectors(const Vector* v1, const Vector* v2, Vector* result);
void transpose_matrix(const Matrix* mat, Matrix* result);

// Utility functions (print, copy, etc.)
void print_matrix(const Matrix* mat);
void print_vector(const Vector* vec);
void copy_matrix(const Matrix* src, Matrix* dest);
void copy_vector(const Vector* src, Vector* dest);

#endif /* NN_MATRIX_H */
