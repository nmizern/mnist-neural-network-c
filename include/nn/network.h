#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include <stddef.h>
#include "matrix.h"
#include "nn_functions.h"

typedef struct {
    size_t num_layers;      // Number of layers
    size_t *layer_sizes;    // Array of layer sizes
    Matrix **layers;        // Array of pointers to layers
} Network;

Network* nn_network_create(const size_t *layer_sizes, size_t num_layers);
void nn_network_free(Network *network);

void nn_network_forward(const Network *network, const Vector *input, Vector *output);
void nn_network_backward(Network *network, const Vector *input, const Vector *target, float learning_rate);
void nn_network_predict(const Network *network, const Vector *input, size_t *predicted_class);

void nn_network_save(const Network *network, const char *filepath);
void nn_network_load(Network *network, const char *filepath);
void nn_network_copy(Network *dest, const Network *src);
void nn_network_print(const Network *network);
void nn_network_zero_gradients(Network *network);

#endif /* NN_NETWORK_H */
