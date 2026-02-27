#ifndef NN_FUNCTIONS_H
#define NN_FUNCTIONS_H

#include <stddef.h>
#include <stdlib.h>
#include "matrix.h"

/* Types énumérés pour la modularité */
typedef enum {
    LOSS_MSE,
} LossType;

typedef enum {
    ACTIVATION_SIGMOID,
    ACTIVATION_RELU,
    ACTIVATION_SOFTMAX
} ActivationType;

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM
} OptimizerType;

/* Optimiseur SGD */
typedef struct {
    float learning_rate;
    float eta;
} SGDOptimizer;

SGDOptimizer* nn_sgd_create(float learning_rate, float eta);
void nn_sgd_free(SGDOptimizer *opt);
void nn_sgd_update(SGDOptimizer *opt, Matrix *weights, const Matrix *gradients);

/* Fonction de perte MSE */
float nn_loss_mse(const Vector *predicted, const Vector *target);
void nn_loss_mse_gradient(const Vector *predicted, const Vector *target, Vector *gradient);

/* Fonctions d'activation */
void nn_sigmoid(const Vector *input, Vector *output);
void nn_relu(const Vector *input, Vector *output);
void nn_softmax(const Vector *input, Vector *output);

/* Dérivées */
void nn_sigmoid_derivative(const Vector *output, Vector *result);
void nn_relu_derivative(const Vector *input, Vector *result);

/* Application générique */
void nn_activation_apply(Vector *vec, ActivationType type);
void nn_activation_derivative(const Vector *vec, Vector *result, ActivationType type);

/* Métriques et utilitaires */
float nn_accuracy(const size_t *predictions, const size_t *targets, size_t count);
size_t nn_argmax(const Vector *vec);
void nn_one_hot(size_t label, size_t num_classes, Vector *result);

#endif // NN_FUNCTIONS_H