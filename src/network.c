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
    Vector *predicted = create_vector(network->layer_sizes[network->num_layers]);
    // Une passe avant pour calculer les activations et les mettre en cache
    nn_network_forward(network, input, predicted);
    

    // Calculer l'erreur de sortie: delta = loss'(predicted, target) * activation'(output)
    Vector *target_one_hot = create_vector(predicted->size);
    nn_one_hot((size_t)target->data[0], predicted->size, target_one_hot);
    Vector *delta = create_vector(predicted->size);
    nn_loss_mse_gradient(predicted, target_one_hot, delta);
    free_vector(target_one_hot);
    
    // Calcul de la dérivée de l'activation pour la dernière couche
    Vector *activation_deriv = create_vector(predicted->size);
    nn_activation_derivative(predicted, activation_deriv, ACTIVATION_SIGMOID);
    
    // Multiplication élément par élément: delta = loss' * activation'
    for (size_t i = 0; i < delta->size; i++) {
        delta->data[i] *= activation_deriv->data[i];
    }
    free_vector(activation_deriv);
    
    // Rétropropagation à travers chaque couche
    for (size_t l = network->num_layers; l > 0; l--) {
        Layer *layer = network->layers[l - 1];
        
        // Créer le vecteur d'entrée avec biais pour le calcul du gradient
        Vector *input_with_bias = create_vector(layer->input_cache->size + 1);
        memcpy(input_with_bias->data, layer->input_cache->data, 
               layer->input_cache->size * sizeof(float));
        input_with_bias->data[layer->input_cache->size] = 1.0f;
        
        // Calculer le gradient pour cette couche: dW = delta * input^T
        for (size_t i = 0; i < layer->weights->rows; i++) {
            for (size_t j = 0; j < layer->weights->cols; j++) {
                layer->gradients->data[i * layer->weights->cols + j] = 
                    delta->data[i] * input_with_bias->data[j];
            }
        }
        
        // Propager l'erreur à la couche précédente (si ce n'est pas la première couche)
        if (l > 1) {
            // Créer le nouveau delta pour la couche précédente
            Vector *next_delta = create_vector(layer->input_cache->size);
            
            // delta_{l-1} = W_l^T * delta_l (sans le biais)
            for (size_t i = 0; i < layer->input_cache->size; i++) {
                next_delta->data[i] = 0.0f;
                for (size_t j = 0; j < layer->weights->rows; j++) {
                    next_delta->data[i] += layer->weights->data[j * layer->weights->cols + i] * delta->data[j];
                }
            }
            
            // Multiplier par la dérivée de l'activation de la couche précédente
            Vector *prev_activation_deriv = create_vector(layer->input_cache->size);
            nn_activation_derivative(layer->input_cache, prev_activation_deriv, ACTIVATION_SIGMOID);
            
            for (size_t i = 0; i < next_delta->size; i++) {
                next_delta->data[i] *= prev_activation_deriv->data[i];
            }
            
            free_vector(prev_activation_deriv);
            free_vector(delta);
            delta = next_delta;
        }
        
        free_vector(input_with_bias);
    }
    
    // Mise à jour des poids: W = W - learning_rate * dW
    for (size_t i = 0; i < network->num_layers; i++) {
        Layer *layer = network->layers[i];
        for (size_t r = 0; r < layer->weights->rows; r++) {
            for (size_t c = 0; c < layer->weights->cols; c++) {
                size_t idx = r * layer->weights->cols + c;
                layer->weights->data[idx] -= learning_rate * layer->gradients->data[idx];
            }
        }
    }

    free_vector(delta);
    free_vector(predicted);
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

/* Sauvegarde et chargement du réseau */
void nn_network_save(const Network *network, const char *filepath) {
    FILE *file = fopen(filepath, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file for saving: %s\n", filepath);
        return;
    }
    
    // Sauvegarder la structure du réseau
    fwrite(&network->num_layers, sizeof(size_t), 1, file);
    fwrite(network->layer_sizes, sizeof(size_t), network->num_layers + 1, file);
    
    // Sauvegarder les poids de chaque couche
    for (size_t i = 0; i < network->num_layers; i++) {
        Layer *layer = network->layers[i];
        fwrite(&layer->weights->rows, sizeof(size_t), 1, file);
        fwrite(&layer->weights->cols, sizeof(size_t), 1, file);
        fwrite(layer->weights->data, sizeof(float), 
               layer->weights->rows * layer->weights->cols, file);
    }
    
    fclose(file);
    printf("Network saved to %s\n", filepath);
}

void nn_network_load(Network *network, const char *filepath) {
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file for loading: %s\n", filepath);
        return;
    }
    
    // Charger et vérifier la structure
    size_t num_layers;
    fread(&num_layers, sizeof(size_t), 1, file);
    if (num_layers != network->num_layers) {
        fprintf(stderr, "Network structure mismatch\n");
        fclose(file);
        return;
    }
    
    size_t *layer_sizes = (size_t*)malloc((num_layers + 1) * sizeof(size_t));
    fread(layer_sizes, sizeof(size_t), num_layers + 1, file);
    
    // Charger les poids
    for (size_t i = 0; i < network->num_layers; i++) {
        size_t rows, cols;
        fread(&rows, sizeof(size_t), 1, file);
        fread(&cols, sizeof(size_t), 1, file);
        fread(network->layers[i]->weights->data, sizeof(float), rows * cols, file);
    }
    
    free(layer_sizes);
    fclose(file);
    printf("Network loaded from %s\n", filepath);
}

void nn_network_copy(Network *dest, const Network *src) {
    if (dest->num_layers != src->num_layers) {
        fprintf(stderr, "Cannot copy networks with different structures\n");
        return;
    }
    
    for (size_t i = 0; i < src->num_layers; i++) {
        copy_matrix(src->layers[i]->weights, dest->layers[i]->weights);
    }
}

/* Affichage du réseau */
void nn_network_print(const Network *network) {
    printf("Network structure:\n");
    printf("  Layers: %zu\n", network->num_layers + 1);
    for (size_t i = 0; i <= network->num_layers; i++) {
        printf("    Layer %zu: %zu neurons\n", i, network->layer_sizes[i]);
    }
}

float nn_network_evaluate(const Network *network, const mnist_dataset_t *dataset) {
    size_t correct = 0;
    for (size_t i = 0; i < dataset->count; i++) {
        Vector *input = create_vector(784);
        for (size_t j = 0; j < 784; j++) {
            input->data[j] = dataset->images[i * 784 + j];
        }

        size_t predicted_class;
        nn_network_predict(network, input, &predicted_class);

        if (predicted_class == dataset->labels[i]) {
            correct++;
        }

        free_vector(input);
    }

    return (float)correct / (float)dataset->count;
}
