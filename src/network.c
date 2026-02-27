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
        network->layers[i]->bias_input = create_vector(layer_sizes[i] + 1);
    }

    size_t max_size = 0;
    for (size_t i = 1; i < num_layers; i++) {
        if (layer_sizes[i] > max_size) {
            max_size = layer_sizes[i];
        }
    }

    size_t output_size = layer_sizes[num_layers - 1];
    network->work_predicted = create_vector(output_size);
    network->work_one_hot = create_vector(output_size);
    network->work_delta = create_vector(max_size);
    network->work_prev_delta = create_vector(max_size);
    network->work_act_deriv = create_vector(max_size);
    
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
            free_vector(network->layers[i]->bias_input);
            free(network->layers[i]);
        }
    }
    
    free_vector(network->work_predicted);
    free_vector(network->work_one_hot);
    free_vector(network->work_delta);
    free_vector(network->work_prev_delta);
    free_vector(network->work_act_deriv);

    free(network->layers);
    free(network->layer_sizes);
    free(network);
}

/* Propagation avant (zero malloc) */
void nn_network_forward(const Network *network, const Vector *input) {
    copy_vector(input, network->layers[0]->input_cache);

    for (size_t i = 0; i < network->num_layers; i++) {
        Layer *layer = network->layers[i];

        memcpy(layer->bias_input->data, layer->input_cache->data, layer->input_cache->size * sizeof(float));
        layer->bias_input->data[layer->input_cache->size] = 1.0f;

        mat_vec_mul(layer->weights, layer->bias_input, layer->output_cache);
        nn_activation_apply(layer->output_cache, ACTIVATION_SIGMOID);

        if (i + 1 < network->num_layers) {
            copy_vector(layer->output_cache, network->layers[i + 1]->input_cache);
        }
    }

    // copy_vector(network->layers[network->num_layers - 1]->output_cache);
}

void nn_network_backward(Network *network, const Vector *input, const Vector *target, float learning_rate) {
    // Une passe avant pour calculer les activations et les mettre en cache
    nn_network_forward(network, input);

    // Calculer l'erreur de sortie: delta = loss'(predicted, target) * activation'(output)
    // Réutilisation des buffers pré-alloués: work_one_hot, work_delta, work_act_deriv
    size_t output_size = network->layers[network->num_layers - 1]->output_cache->size;
    nn_one_hot((size_t)target->data[0], output_size, network->work_one_hot);
    nn_loss_mse_gradient(network->layers[network->num_layers - 1]->output_cache, network->work_one_hot, network->work_delta);

    // Calcul de la dérivée de l'activation pour la dernière couche (réutilise work_act_deriv)
    nn_activation_derivative(network->layers[network->num_layers - 1]->output_cache, network->work_act_deriv, ACTIVATION_SIGMOID);

    // Multiplication élément par élément: delta = loss' * activation'
    for (size_t i = 0; i < output_size; i++) {
        network->work_delta->data[i] *= network->work_act_deriv->data[i];
    }

    // Ping-pong entre work_delta et work_prev_delta pour éviter tout malloc dans la boucle
    Vector *delta      = network->work_delta;
    Vector *next_delta = network->work_prev_delta;

    // Rétropropagation à travers chaque couche
    for (size_t l = network->num_layers; l > 0; l--) {
        Layer *layer = network->layers[l - 1];

        // layer->bias_input est déjà rempli par la passe avant: [input_cache | 1.0]
        // On l'utilise directement pour le calcul du gradient: dW = delta * input^T
        for (size_t i = 0; i < layer->weights->rows; i++) {
            for (size_t j = 0; j < layer->weights->cols; j++) {
                layer->gradients->data[i * layer->weights->cols + j] =
                    delta->data[i] * layer->bias_input->data[j];
            }
        }

        // Propager l'erreur à la couche précédente (si ce n'est pas la première couche)
        if (l > 1) {
            size_t prev_size = layer->input_cache->size;

            // delta_{l-1} = W_l^T * delta_l (sans la colonne biais)
            for (size_t i = 0; i < prev_size; i++) {
                next_delta->data[i] = 0.0f;
                for (size_t j = 0; j < layer->weights->rows; j++) {
                    next_delta->data[i] += layer->weights->data[j * layer->weights->cols + i] * delta->data[j];
                }
            }

            // Multiplier par la dérivée de l'activation de la couche précédente (réutilise work_act_deriv)
            nn_activation_derivative(layer->input_cache, network->work_act_deriv, ACTIVATION_SIGMOID);
            for (size_t i = 0; i < prev_size; i++) {
                next_delta->data[i] *= network->work_act_deriv->data[i];
            }

            // Échange des pointeurs (ping-pong), aucun malloc/free
            Vector *tmp = delta;
            delta       = next_delta;
            next_delta  = tmp;
        }
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
    // Aucun free nécessaire
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
    nn_network_forward(network, input);
    *predicted_class = nn_argmax(network->layers[network->num_layers - 1]->output_cache);
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
    Vector *input = create_vector(784);
    for (size_t i = 0; i < dataset->count; i++) {
        for (size_t j = 0; j < 784; j++) {
            input->data[j] = dataset->images[i * 784 + j];
        }

        size_t predicted_class;
        nn_network_predict(network, input, &predicted_class);

        if (predicted_class == dataset->labels[i]) {
            correct++;
        }
    }
    free_vector(input);

    return (float)correct / (float)dataset->count;
}

void nn_network_train(Network *network, mnist_dataset_t *train_data, float learning_rate, int epochs) {
    Vector *input  = create_vector(784);
    Vector *target = create_vector(1);
    for (int i = 0; i < epochs; i++) {
        mnist_shuffle(train_data);
        for (size_t j = 0; j < train_data->count; j++) {
            for (size_t k = 0; k < 784; k++) {
                input->data[k] = train_data->images[j * 784 + k];
            }
            target->data[0] = train_data->labels[j];

            nn_network_backward(network, input, target, learning_rate);
        }
        printf("Epoch %d/%d complete\n", i+1, epochs);
    }
    free_vector(input);
    free_vector(target);
}
