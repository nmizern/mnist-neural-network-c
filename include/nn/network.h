#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include <stddef.h>
#include "matrix.h"
#include "nn_functions.h"
#include "mnist.h"

typedef struct {
    Matrix *weights;
    Vector *input_cache;
    Vector *output_cache;
    Vector *bias_input;
} Layer;

typedef struct {
    size_t num_layers;
    size_t *layer_sizes;
    Layer **layers;

    // Configuration d'entrainement
    ActivationType hidden_activation;
    ActivationType output_activation;
    LossType loss_type;
    OptimizerType optimizer_type;

    // État de l'optimiseur adaptatif (alloué uniquement si nécessaire)
    AdamOptimizer **adam_opts;
    Matrix **opt_gradients;

    // Tampons de travail pré-alloués (zéro malloc pendant l'entrainement)
    Vector *work_predicted;
    Vector *work_one_hot;
    Vector *work_delta;
    Vector *work_prev_delta;
    Vector *work_act_deriv;
} Network;

Network* nn_network_create(const size_t *layer_sizes, size_t num_layers);
void nn_network_free(Network *network);

void nn_network_forward(const Network *network, const Vector *input);
void nn_network_backward(Network *network, const Vector *input, const Vector *target, float learning_rate);

void nn_network_set_training_config(Network *network,
                                    ActivationType hidden_activation,
                                    ActivationType output_activation,
                                    LossType loss_type,
                                    OptimizerType optimizer_type);

void nn_network_predict(const Network *network, const Vector *input, size_t *predicted_class);

void nn_network_save(const Network *network, const char *filepath);
void nn_network_load(Network *network, const char *filepath);
void nn_network_copy(Network *dest, const Network *src);
void nn_network_print(const Network *network);

void nn_network_zero_gradients(Network *network);

float nn_network_evaluate(const Network *network, const mnist_dataset_t *dataset);
void nn_network_train(Network *network, mnist_dataset_t *train_data, float learning_rate, int epochs);

#endif // NN_NETWORK_H
