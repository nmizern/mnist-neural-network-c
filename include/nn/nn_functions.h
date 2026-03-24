#ifndef NN_FUNCTIONS_H
#define NN_FUNCTIONS_H

#include <stddef.h>
#include <stdlib.h>
#include "matrix.h"

/* Énumérations */
typedef enum {
    LOSS_MSE,
    LOSS_MAE,
    LOSS_BINARY_CROSS_ENTROPY,
    LOSS_CATEGORICAL_CROSS_ENTROPY
} LossType;

typedef enum {
    ACTIVATION_SIGMOID,
    ACTIVATION_RELU,
    ACTIVATION_SOFTMAX,
    ACTIVATION_TANH,
    ACTIVATION_LEAKY_RELU,
    ACTIVATION_LINEAR
} ActivationType;

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM,
    OPTIMIZER_SGD_MOMENTUM,
    OPTIMIZER_RMSPROP
} OptimizerType;

/* Optimiseurs */
typedef struct {
    float learning_rate;
    float eta;
} SGDOptimizer;

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    size_t timestep;
    Matrix *m;
    Matrix *v;
} AdamOptimizer;

SGDOptimizer* nn_sgd_create(float learning_rate, float eta);
void nn_sgd_free(SGDOptimizer *opt);
void nn_sgd_update(SGDOptimizer *opt, Matrix *weights, const Matrix *gradients);

AdamOptimizer* nn_adam_create(size_t rows, size_t cols, float learning_rate, float beta1, float beta2, float epsilon);
void nn_adam_free(AdamOptimizer *opt);
void nn_adam_update(AdamOptimizer *opt, Matrix *weights, const Matrix *gradients);

/* Fonctions de perte */
float nn_loss_mse(const Vector *predicted, const Vector *target);
void nn_loss_mse_gradient(const Vector *predicted, const Vector *target, Vector *gradient);
float nn_loss_mae(const Vector *predicted, const Vector *target);
void nn_loss_mae_gradient(const Vector *predicted, const Vector *target, Vector *gradient);
float nn_loss_binary_cross_entropy(const Vector *predicted, const Vector *target);
void nn_loss_binary_cross_entropy_gradient(const Vector *predicted, const Vector *target, Vector *gradient);
float nn_loss_categorical_cross_entropy(const Vector *predicted, const Vector *target);
void nn_loss_categorical_cross_entropy_gradient(const Vector *predicted, const Vector *target, Vector *gradient);
float nn_loss_compute(const Vector *predicted, const Vector *target, LossType type);
void nn_loss_gradient(const Vector *predicted, const Vector *target, Vector *gradient, LossType type);

/* Fonctions d'activation */
void nn_sigmoid(const Vector *input, Vector *output);
void nn_relu(const Vector *input, Vector *output);
void nn_softmax(const Vector *input, Vector *output);
void nn_tanh(const Vector *input, Vector *output);
void nn_leaky_relu(const Vector *input, Vector *output, float alpha);
void nn_linear(const Vector *input, Vector *output);

/* Dérivées des fonctions d'activation */
void nn_sigmoid_derivative(const Vector *output, Vector *result);
void nn_relu_derivative(const Vector *input, Vector *result);
void nn_softmax_derivative(const Vector *output, Vector *result);
void nn_tanh_derivative(const Vector *output, Vector *result);
void nn_leaky_relu_derivative(const Vector *input, Vector *result, float alpha);
void nn_linear_derivative(const Vector *input, Vector *result);

/* Application générique des activations */
void nn_activation_apply(Vector *vec, ActivationType type);
void nn_activation_derivative(const Vector *vec, Vector *result, ActivationType type);

/* Métriques et utilitaires */
float nn_accuracy(const size_t *predictions, const size_t *targets, size_t count);
size_t nn_argmax(const Vector *vec);
void nn_one_hot(size_t label, size_t num_classes, Vector *result);

#endif // NN_FUNCTIONS_H
