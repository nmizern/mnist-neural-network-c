#include "nn/nn_functions.h"
#include <math.h>
#include <string.h>

static float nn_clamp(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

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

void nn_tanh(const Vector *input, Vector *output) {
    for (size_t i = 0; i < input->size; i++) {
        output->data[i] = tanhf(input->data[i]);
    }
}

void nn_leaky_relu(const Vector *input, Vector *output, float alpha) {
    for (size_t i = 0; i < input->size; i++) {
        output->data[i] = input->data[i] > 0.0f ? input->data[i] : alpha * input->data[i];
    }
}

void nn_linear(const Vector *input, Vector *output) {
    copy_vector(input, output);
}

/* Dérivées des fonctions d'activation */
void nn_sigmoid_derivative(const Vector *output, Vector *result) {
    for (size_t i = 0; i < output->size; i++) {
        result->data[i] = output->data[i] * (1.0f - output->data[i]);
    }
}

void nn_relu_derivative(const Vector *input, Vector *result) {
    for (size_t i = 0; i < input->size; i++) {
        result->data[i] = input->data[i] > 0.0f ? 1.0f : 0.0f;
    }
}

void nn_softmax_derivative(const Vector *output, Vector *result) {
    for (size_t i = 0; i < output->size; i++) {
        result->data[i] = output->data[i] * (1.0f - output->data[i]);
    }
}

void nn_tanh_derivative(const Vector *output, Vector *result) {
    for (size_t i = 0; i < output->size; i++) {
        result->data[i] = 1.0f - output->data[i] * output->data[i];
    }
}

void nn_leaky_relu_derivative(const Vector *input, Vector *result, float alpha) {
    for (size_t i = 0; i < input->size; i++) {
        result->data[i] = input->data[i] > 0.0f ? 1.0f : alpha;
    }
}

void nn_linear_derivative(const Vector *input, Vector *result) {
    (void)input;
    for (size_t i = 0; i < result->size; i++) {
        result->data[i] = 1.0f;
    }
}

/* Application des activations selon le type */
void nn_activation_apply(Vector *vec, ActivationType type) {
      switch (type) {
          case ACTIVATION_SIGMOID:
              nn_sigmoid(vec, vec);
              break;
          case ACTIVATION_RELU:
              nn_relu(vec, vec);
              break;
          case ACTIVATION_SOFTMAX:
              nn_softmax(vec, vec);
              break;
          case ACTIVATION_TANH:
              nn_tanh(vec, vec);
              break;
          case ACTIVATION_LEAKY_RELU:
              nn_leaky_relu(vec, vec, 0.01f);
              break;
          case ACTIVATION_LINEAR:
              nn_linear(vec, vec);
              break;
      }
}

void nn_activation_derivative(const Vector *vec, Vector *result, ActivationType type) {
    switch (type) {
        case ACTIVATION_SIGMOID:
            nn_sigmoid_derivative(vec, result);
            break;
        case ACTIVATION_RELU:
            nn_relu_derivative(vec, result);
            break;
        case ACTIVATION_SOFTMAX:
            nn_softmax_derivative(vec, result);
            break;
        case ACTIVATION_TANH:
            nn_tanh_derivative(vec, result);
            break;
        case ACTIVATION_LEAKY_RELU:
            nn_leaky_relu_derivative(vec, result, 0.01f);
            break;
        case ACTIVATION_LINEAR:
            nn_linear_derivative(vec, result);
            break;
        default:
            break;
    }
}

// Fonction de perte MSE (erreur quadratique moyenne)
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

float nn_loss_mae(const Vector *predicted, const Vector *target) {
    float sum = 0.0f;
    for (size_t i = 0; i < predicted->size; i++) {
        sum += fabsf(predicted->data[i] - target->data[i]);
    }
    return sum / (float)predicted->size;
}

void nn_loss_mae_gradient(const Vector *predicted, const Vector *target, Vector *gradient) {
    for (size_t i = 0; i < predicted->size; i++) {
        float diff = predicted->data[i] - target->data[i];
        if (diff > 0.0f) {
            gradient->data[i] = 1.0f / (float)predicted->size;
        } else if (diff < 0.0f) {
            gradient->data[i] = -1.0f / (float)predicted->size;
        } else {
            gradient->data[i] = 0.0f;
        }
    }
}

float nn_loss_binary_cross_entropy(const Vector *predicted, const Vector *target) {
    const float eps = 1e-7f;
    float sum = 0.0f;
    for (size_t i = 0; i < predicted->size; i++) {
        float p = nn_clamp(predicted->data[i], eps, 1.0f - eps);
        float t = target->data[i];
        sum += -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
    }
    return sum / (float)predicted->size;
}

void nn_loss_binary_cross_entropy_gradient(const Vector *predicted, const Vector *target, Vector *gradient) {
    const float eps = 1e-7f;
    for (size_t i = 0; i < predicted->size; i++) {
        float p = nn_clamp(predicted->data[i], eps, 1.0f - eps);
        float t = target->data[i];
        gradient->data[i] = (p - t) / (p * (1.0f - p) * (float)predicted->size);
    }
}

float nn_loss_categorical_cross_entropy(const Vector *predicted, const Vector *target) {
    const float eps = 1e-7f;
    float sum = 0.0f;
    for (size_t i = 0; i < predicted->size; i++) {
        float p = nn_clamp(predicted->data[i], eps, 1.0f);
        sum += -target->data[i] * logf(p);
    }
    return sum / (float)predicted->size;
}

void nn_loss_categorical_cross_entropy_gradient(const Vector *predicted, const Vector *target, Vector *gradient) {
    const float eps = 1e-7f;
    for (size_t i = 0; i < predicted->size; i++) {
        float p = nn_clamp(predicted->data[i], eps, 1.0f);
        gradient->data[i] = -(target->data[i] / p) / (float)predicted->size;
    }
}

float nn_loss_compute(const Vector *predicted, const Vector *target, LossType type) {
    switch (type) {
        case LOSS_MSE:
            return nn_loss_mse(predicted, target);
        case LOSS_MAE:
            return nn_loss_mae(predicted, target);
        case LOSS_BINARY_CROSS_ENTROPY:
            return nn_loss_binary_cross_entropy(predicted, target);
        case LOSS_CATEGORICAL_CROSS_ENTROPY:
            return nn_loss_categorical_cross_entropy(predicted, target);
        default:
            return nn_loss_mse(predicted, target);
    }
}

void nn_loss_gradient(const Vector *predicted, const Vector *target, Vector *gradient, LossType type) {
    switch (type) {
        case LOSS_MSE:
            nn_loss_mse_gradient(predicted, target, gradient);
            break;
        case LOSS_MAE:
            nn_loss_mae_gradient(predicted, target, gradient);
            break;
        case LOSS_BINARY_CROSS_ENTROPY:
            nn_loss_binary_cross_entropy_gradient(predicted, target, gradient);
            break;
        case LOSS_CATEGORICAL_CROSS_ENTROPY:
            nn_loss_categorical_cross_entropy_gradient(predicted, target, gradient);
            break;
        default:
            nn_loss_mse_gradient(predicted, target, gradient);
            break;
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

AdamOptimizer* nn_adam_create(size_t rows, size_t cols, float learning_rate, float beta1, float beta2, float epsilon) {
    AdamOptimizer *opt = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
    opt->learning_rate = learning_rate;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->epsilon = epsilon;
    opt->timestep = 0;
    opt->m = create_matrix(rows, cols);
    opt->v = create_matrix(rows, cols);
    return opt;
}

void nn_adam_free(AdamOptimizer *opt) {
    if (!opt) return;
    free_matrix(opt->m);
    free_matrix(opt->v);
    free(opt);
}

void nn_adam_update(AdamOptimizer *opt, Matrix *weights, const Matrix *gradients) {
    opt->timestep++;

    float beta1_t = powf(opt->beta1, (float)opt->timestep);
    float beta2_t = powf(opt->beta2, (float)opt->timestep);
    float bias_corr1 = 1.0f - beta1_t;
    float bias_corr2 = 1.0f - beta2_t;

    for (size_t i = 0; i < weights->rows * weights->cols; i++) {
        float g = gradients->data[i];
        opt->m->data[i] = opt->beta1 * opt->m->data[i] + (1.0f - opt->beta1) * g;
        opt->v->data[i] = opt->beta2 * opt->v->data[i] + (1.0f - opt->beta2) * g * g;

        float m_hat = opt->m->data[i] / bias_corr1;
        float v_hat = opt->v->data[i] / bias_corr2;
        weights->data[i] -= opt->learning_rate * (m_hat / (sqrtf(v_hat) + opt->epsilon));
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
