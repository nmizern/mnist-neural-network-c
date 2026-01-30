

#include <stddef.h>

#include "matrix.h"

/* Loss function types */
typedef enum {
    LOSS_MSE,           /* Mean Squared Error */
    LOSS_CROSS_ENTROPY  /* Cross Entropy */
} LossType;

typedef enum {
    ACTIVATION_SIGMOID,
    ACTIVATION_RELU,
    ACTIVATION_TANH,
    ACTIVATION_SOFTMAX
} ActivationType;

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM
} OptimizerType;

typedef struct {
    float learning_rate;
    float momentum;
} SGDOptimizer;

// 
// Optimizers
// 

SGDOptimizer* nn_sgd_create(float learning_rate, float momentum);
void nn_sgd_free(SGDOptimizer *opt);

void nn_sgd_update(SGDOptimizer *opt, Matrix *weights, const Matrix *gradients);
void nn_sgd_update_vector(SGDOptimizer *opt, Vector *biases, const Vector *gradients);

// 
// Loss
// 
// Compute loss value
float nn_loss_mse(const Vector *predicted, const Vector *target);
float nn_loss_cross_entropy(const Vector *predicted, const Vector *target);

// Compute loss gradient for backpropagation
void nn_loss_mse_gradient(const Vector *predicted, const Vector *target, Vector *gradient);
void nn_loss_cross_entropy_gradient(const Vector *predicted, const Vector *target, Vector *gradient);

// 
// Activation functions
//

void nn_sigmoid(const Vector *input, Vector *output);
void nn_relu(const Vector *input, Vector *output);
void nn_softmax(const Vector *input, Vector *output);

void nn_sigmoid_derivative(const Vector *output, Vector *result);
void nn_relu_derivative(const Vector *input, Vector *result);

// Apply activation
void nn_activation_apply(Vector *vec, ActivationType type);
void nn_activation_derivative(const Vector *vec, Vector *result, ActivationType type);

// Metric functions
float nn_accuracy(const size_t *predictions, const size_t *targets, size_t count);
size_t nn_argmax(const Vector *vec);
