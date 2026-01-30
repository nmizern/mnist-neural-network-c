#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include <stddef.h>
#include "matrix.h"
#include "mnist.h"

typedef struct {
    matrix_t *W;        
    float    *B;        
    float    *z;        
    float    *a;        
    size_t    in_size;
    size_t    out_size;
} layer_t;


typedef struct {
    layer_t *layers;                            
    size_t   num_layers;  
    size_t  *sizes;       
    size_t   num_sizes;
} network_t;

network_t *network_create(const size_t *sizes, size_t num_sizes);

void network_destroy(network_t *net);

float *network_forward(network_t *net, const float *input);

int network_predict(network_t *net, const float *input);

void network_train(network_t *net, mnist_dataset_t *train_data,
                   float learning_rate, int epochs);

float network_accuracy(network_t *net, mnist_dataset_t *dataset);

#endif 
