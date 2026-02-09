#ifndef NN_MATRIX_H
#define NN_MATRIX_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


// Cette structure de matrice sera utilisée pour définir les matrices dans le réseau de neurones
// Ces matrices représenteront les poids, les biais, et les couches ou 'layers' du réseau
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


// Création et destruction de matrice
Matrix* create_matrix(size_t rows, size_t cols);
Matrix* create_identity_matrix(size_t rows, size_t cols);
Matrix* create_rndm_matrix(size_t rows, size_t cols);
void free_matrix(Matrix* mat);

Vector* create_vector(size_t size);
Vector* create_rndm_vector(size_t size); // Utilisé pour initialiser les biais aléatoirement
void free_vector(Vector* vec);


// Multiplication matrice-vecteur
// On multiplie la matrice mat (de taille rows x cols) avec le vecteur vec (de taille cols x 1)
// Le résultat est stocké dans le vecteur result (de taille rows x 1)
void mat_vec_mul(const Matrix* mat, Vector* vec, Vector* result);
void mat_mat_mul(const Matrix* A, const Matrix* B, Matrix* result);

// Opérations élémentaires
float dot_product(const Vector* v1, const Vector* v2);
Vector* add_vectors(const Vector* v1, const Vector* v2);
Vector* multiply_vectors(const Vector* v1, const Vector* v2);
Vector* divide_vectors(const Vector* v1, const Vector* v2);
Vector* subtract_vectors(const Vector* v1, const Vector* v2);
Matrix* transpose_matrix(const Matrix* mat);

// Fonctions utilitaires (affichage, copie, etc.)
void print_matrix(const Matrix* mat);
void print_matrix_dimentions(const Matrix* mat);
void print_vector(const Vector* vec);
void copy_matrix(const Matrix* src, Matrix* dest);
void copy_vector(const Vector* src, Vector* dest);

#endif // NN_MATRIX_H