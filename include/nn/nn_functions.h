#include <stddef.h>
#include <stdlib.h>
#include "matrix.h"


/*
En C, les énumérations sont un type de données défini par l'utilisateur,
constitué d'un ensemble de constantes entières nommées, appelées énumérateurs.
Elles améliorent la lisibilité et la maintenabilité du code, et permettent d'avoir un système
modulaire qui est justement ce que l'on cherche.
*/
typedef enum {
    LOSS_MSE,
} LossType;

typedef enum {
    ACTIVATION_SIGMOID,
    ACTIVATION_RELU,
    ACTIVATION_SOFTMAX // Dans l'énoncé n'a pas été donné mais c'est une fonction d'activation de base.
                       // Utile pour d'autres réseaux simples.
} ActivationType;

typedef enum {
    OPTIMIZER_SGD, // Stochastic Gradient Descent.
    OPTIMIZER_ADAM // Pas demandé dans l'énoncé mais utile et comme pour le softmax, pas obligé de l'implémenter.
} OptimizerType;

// L'utilisation des enums fait que l'on n'est pas obligé d'implémenter les autres fonctions et on peut
// se concentrer sur les fonctions qui nous sont proposées pour l'instant et puis on peut rajouter d'autres fonctions

// 
// Optimiseurs
// 
typedef struct {
    float learning_rate;
    float eta; // correspond au eta qui est présent dans l'algorithme de 'Stochastic Gradient Descent'
} SGDOptimizer;

SGDOptimizer* nn_sgd_create(float learning_rate, float eta);
void nn_sgd_free(SGDOptimizer *opt);
void nn_sgd_update(SGDOptimizer *opt, Matrix *weights, const Matrix *gradients);    
// Les poids et biais sont dans une même matrice assimilée dans cette fonction à la matrice weights

// 
// Fonction de perte
// 
// Calcul de la perte
float nn_loss_mse(const Vector *predicted, const Vector *target);

// Calcul du gradient de la perte
void nn_loss_mse_gradient(const Vector *predicted, const Vector *target, Vector *gradient);

// 
// Fonctions d'activation
//
void nn_sigmoid(const Vector *input, Vector *output);
void nn_relu(const Vector *input, Vector *output);
void nn_softmax(const Vector *input, Vector *output);

void nn_sigmoid_derivative(const Vector *output, Vector *result);
void nn_relu_derivative(const Vector *input, Vector *result);

// Appliquer l'activation
void nn_activation_apply(Vector *vec, ActivationType type);
void nn_activation_derivative(const Vector *vec, Vector *result, ActivationType type);

// Fonctions de métrique
float nn_accuracy(const size_t *predictions, const size_t *targets, size_t count);
size_t nn_argmax(const Vector *vec);