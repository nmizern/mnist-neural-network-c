#include "nn/nn_functions.h"
#include <math.h>
#include <string.h>

/* Fonctions d'activation */
void nn_sigmoid(const Vector *input, Vector *output) {
    for (size_t i = 0; i < input->size; i++) {
        float x = input->data[i];
        if (x < -500.0f) x = -500.0f;
        if (x > 500.0f) x = 500.0f;
        output->data[i] = 1.0f / (1.0f + expf(-x));
    }
}

void nn_relu(const Vector *input, Vector *output) {
    for (size_t i = 0; i < input->size; i++) {
        output->data[i] = input->data[i] > 0.0f ? input->data[i] : 0.0f;
    }
}

void nn_softmax(const Vector *input, Vector *output) {
    float max_val = input->data[0];
    for (size_t i = 1; i < input->size; i++) {
        if (input->data[i] > max_val) max_val = input->data[i];
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < input->size; i++) {
        output->data[i] = expf(input->data[i] - max_val);
        sum += output->data[i];
    }
    
    for (size_t i = 0; i < input->size; i++) {
        output->data[i] /= sum;
    }
}

// Derivées des fonctions d'activations
void nn_sigmoid_derivative(const Vector *output, Vector *result) {
    for (size_t i = 0; i < output->size; i++) {
        result->data[i] = output->data[i] * (1.0f - output->data[i]);
    }
}
// On utilise l'expression de l'équation differentielle de la fonction sigmoïde pour calculer sa dérivée

// ReLU n'as pas de dérivée continue et explicite, on utilise une approximation simple
void nn_relu_derivative(const Vector *input, Vector *result) {
    for (size_t i = 0; i < input->size; i++) {
        result->data[i] = input->data[i] > 0.0f ? 1.0f : 0.0f;
    }
}

/* Application des activations selon le type */
void nn_activation_apply(Vector *vec, ActivationType type) {
    Vector *temp = create_vector(vec->size);
    copy_vector(vec, temp);
    
    switch (type) {
        case ACTIVATION_SIGMOID:
            nn_sigmoid(temp, vec);
            break;
        case ACTIVATION_RELU:
            nn_relu(temp, vec);
            break;
        case ACTIVATION_SOFTMAX:
            nn_softmax(temp, vec);
            break;
    }
    
    free_vector(temp);
}

void nn_activation_derivative(const Vector *vec, Vector *result, ActivationType type) {
    switch (type) {
        case ACTIVATION_SIGMOID:
            nn_sigmoid_derivative(vec, result);
            break;
        case ACTIVATION_RELU:
            nn_relu_derivative(vec, result);
            break;
        default:
            break;
    }
}

// Fonction de perte MSE
// mean squared error, L2 norme (squared)
float nn_loss_mse(const Vector *predicted, const Vector *target) {
    float sum = 0.0f;
    for (size_t i = 0; i < predicted->size; i++) {
        float diff = predicted->data[i] - target->data[i];
        sum += diff * diff;
    }
    return sum / (float)predicted->size;
}

void nn_loss_mse_gradient(const Vector *predicted, const Vector *target, Vector *gradient) {
    for (size_t i = 0; i < predicted->size; i++) {
        gradient->data[i] = 2.0f * (predicted->data[i] - target->data[i]) / (float)predicted->size;
    }
}

/* Optimiseur SGD */
SGDOptimizer* nn_sgd_create(float learning_rate, float eta) {
    SGDOptimizer *opt = (SGDOptimizer*)malloc(sizeof(SGDOptimizer));
    opt->learning_rate = learning_rate;
    opt->eta = eta;
    return opt;
}

void nn_sgd_free(SGDOptimizer *opt) {
    free(opt);
}

void nn_sgd_update(SGDOptimizer *opt, Matrix *weights, const Matrix *gradients) {
    for (size_t i = 0; i < weights->rows * weights->cols; i++) {
        weights->data[i] -= opt->learning_rate * gradients->data[i];
    }
}

size_t nn_argmax(const Vector *vec) {
    size_t max_idx = 0;
    float max_val = vec->data[0];
    for (size_t i = 1; i < vec->size; i++) {
        if (vec->data[i] > max_val) {
            max_val = vec->data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

float nn_accuracy(const size_t *predictions, const size_t *targets, size_t count) {
    size_t correct = 0;
    for (size_t i = 0; i < count; i++) {
        if (predictions[i] == targets[i]) {
            correct++;
        }
    }
    return (float)correct / (float)count;
}

void nn_one_hot(size_t label, size_t num_classes, Vector *result) {
    for (size_t i = 0; i < num_classes; i++) {
        if (i == label) {
            result->data[i] = 1.0f;
            continue;
        }
        result->data[i] = 0.0f;
    }
}
