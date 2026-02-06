#include "nn/network.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Création et destruction du réseau
Network* nn_network_create(const size_t *layer_sizes, size_t num_layers) {
    Network *network = (Network*)malloc(sizeof(Network));
    network->num_layers = num_layers - 1;
    network->layer_sizes = (size_t*)malloc(num_layers * sizeof(size_t));
    memcpy(network->layer_sizes, layer_sizes, num_layers * sizeof(size_t));
    
    network->layers = (Layer**)malloc(network->num_layers * sizeof(Layer*));
    
    for (size_t i = 0; i < network->num_layers; i++) {
        network->layers[i] = (Layer*)malloc(sizeof(Layer));
        network->layers[i]->weights = create_rndm_matrix(layer_sizes[i+1], layer_sizes[i] + 1);
        network->layers[i]->gradients = create_matrix(layer_sizes[i+1], layer_sizes[i] + 1);
        network->layers[i]->input_cache = create_vector(layer_sizes[i]);
        network->layers[i]->output_cache = create_vector(layer_sizes[i+1]);
    }
    
    return network;
}

void nn_network_free(Network *network) {
    if (!network) return;
    
    for (size_t i = 0; i < network->num_layers; i++) {
        if (network->layers[i]) {
            free_matrix(network->layers[i]->weights);
            free_matrix(network->layers[i]->gradients);
            free_vector(network->layers[i]->input_cache);
            free_vector(network->layers[i]->output_cache);
            free(network->layers[i]);
        }
    }
    
    free(network->layers);
    free(network->layer_sizes);
    free(network);
}

/* Propagation avant */
void nn_network_forward(const Network *network, const Vector *input, Vector *output) {
    Vector *current = create_vector(input->size);
    copy_vector(input, current);
    
    for (size_t i = 0; i < network->num_layers; i++) {
        Layer *layer = network->layers[i];
        copy_vector(current, layer->input_cache);
        // a chaque etape on copie notre input dans le cache d'entrée de la couche pour pouvoir l'utiliser lors de la rétropropagation
        
        Vector *current_with_bias = create_vector(current->size + 1);
        memcpy(current_with_bias->data, current->data, current->size * sizeof(float));
        current_with_bias->data[current->size] = 1.0f;
        // On ajoute un biais de 1 à la fin du vecteur d'entrée pour permettre le calcul du biais dans la multiplication matrice-vecteur
        
        Vector *result = create_vector(layer->weights->rows); // Vect de sortie
        mat_vec_mul(layer->weights, current_with_bias, result);
        
        nn_activation_apply(result, ACTIVATION_SIGMOID);
        
        free_vector(current);
        free_vector(current_with_bias);
        current = result;
        copy_vector(current, layer->output_cache);
    }
    
    copy_vector(current, output);
    free_vector(current);
}

void nn_network_backward(Network *network, const Vector *input, const Vector *target, float learning_rate) {

}

/* Remise à zéro des gradients */
void nn_network_zero_gradients(Network *network) {
    for (size_t i = 0; i < network->num_layers; i++) {
        memset(network->layers[i]->gradients->data, 0, 
               network->layers[i]->gradients->rows * network->layers[i]->gradients->cols * sizeof(float));
    }
}

/* Prédiction */
void nn_network_predict(const Network *network, const Vector *input, size_t *predicted_class) {
    Vector *output = create_vector(network->layer_sizes[network->num_layers]);
    nn_network_forward(network, input, output);
    *predicted_class = nn_argmax(output);
    free_vector(output);
}

/* Affichage du réseau */
void nn_network_print(const Network *network) {
    printf("Network structure:\n");
    printf("  Layers: %zu\n", network->num_layers + 1);
    for (size_t i = 0; i <= network->num_layers; i++) {
        printf("    Layer %zu: %zu neurons\n", i, network->layer_sizes[i]);
    }
}
