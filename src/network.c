#include "nn/network.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

static void nn_network_clear_optimizer_state(Network *network) {
    if (!network) return;

    if (network->adam_opts) {
        for (size_t i = 0; i < network->num_layers; i++) {
            if (network->adam_opts[i]) {
                nn_adam_free(network->adam_opts[i]);
            }
        }
        free(network->adam_opts);
        network->adam_opts = NULL;
    }

    if (network->opt_gradients) {
        for (size_t i = 0; i < network->num_layers; i++) {
            if (network->opt_gradients[i]) {
                free_matrix(network->opt_gradients[i]);
            }
        }
        free(network->opt_gradients);
        network->opt_gradients = NULL;
    }
}

void nn_network_set_training_config(Network *network,
                                    ActivationType hidden_activation,
                                    ActivationType output_activation,
                                    LossType loss_type,
                                    OptimizerType optimizer_type) {
    if (!network) return;

    // Reinitialise proprement l'etat de l'optimiseur avant d'appliquer une nouvelle config.
    nn_network_clear_optimizer_state(network);

    network->hidden_activation = hidden_activation;
    network->output_activation = output_activation;
    network->loss_type = loss_type;
    network->optimizer_type = optimizer_type;

    if (optimizer_type == OPTIMIZER_ADAM) {
        // Alloue l'etat Adam par couche pour conserver les moments m et v.
        network->adam_opts = (AdamOptimizer**)calloc(network->num_layers, sizeof(AdamOptimizer*));
        network->opt_gradients = (Matrix**)calloc(network->num_layers, sizeof(Matrix*));

        for (size_t i = 0; i < network->num_layers; i++) {
            const size_t rows = network->layers[i]->weights->rows;
            const size_t cols = network->layers[i]->weights->cols;
            network->adam_opts[i] = nn_adam_create(rows, cols, 0.001f, 0.9f, 0.999f, 1e-8f);
            network->opt_gradients[i] = create_matrix(rows, cols);
        }
    }
}

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

    network->hidden_activation = ACTIVATION_SIGMOID;
    network->output_activation = ACTIVATION_SIGMOID;
    network->loss_type = LOSS_MSE;
    network->optimizer_type = OPTIMIZER_SGD;
    network->adam_opts = NULL;
    network->opt_gradients = NULL;
    
    return network;
}

void nn_network_free(Network *network) {
    if (!network) return;
    
    for (size_t i = 0; i < network->num_layers; i++) {
        if (network->layers[i]) {
            free_matrix(network->layers[i]->weights);
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

    nn_network_clear_optimizer_state(network);

    free(network->layers);
    free(network->layer_sizes);
    free(network);
}

/* Propagation avant (zero malloc) */
void nn_network_forward(const Network *network, const Vector *input) {
    // La premiere couche lit directement l'entree utilisateur.
    copy_vector(input, network->layers[0]->input_cache);

    for (size_t i = 0; i < network->num_layers; i++) {
        Layer *layer = network->layers[i];

        memcpy(layer->bias_input->data, layer->input_cache->data, layer->input_cache->size * sizeof(float));
        layer->bias_input->data[layer->input_cache->size] = 1.0f;

        // Produit matrice-vecteur puis activation selon le type de couche.
        mat_vec_mul(layer->weights, layer->bias_input, layer->output_cache);
        if (i + 1 < network->num_layers) {
            nn_activation_apply(layer->output_cache, network->hidden_activation);
        } else {
            nn_activation_apply(layer->output_cache, network->output_activation);
        }

        if (i + 1 < network->num_layers) {
            // La sortie courante devient l'entree de la couche suivante.
            copy_vector(layer->output_cache, network->layers[i + 1]->input_cache);
        }
    }

}

void nn_network_backward(Network *network, const Vector *input, const Vector *target, float learning_rate) {
    // Une passe avant pour calculer les activations et les mettre en cache
    nn_network_forward(network, input);

    // Calculer l'erreur de sortie: delta = loss'(predicted, target) * activation'(output)
    // Réutilisation des buffers pré-alloués: work_one_hot, work_delta, work_act_deriv
    size_t output_size = network->layers[network->num_layers - 1]->output_cache->size;
    const Vector *target_vec = target;
    if (target->size == 1) {
        // Convertit un label scalaire en one-hot pour les pertes multi-classes.
        nn_one_hot((size_t)target->data[0], output_size, network->work_one_hot);
        target_vec = network->work_one_hot;
    }

    nn_loss_gradient(network->layers[network->num_layers - 1]->output_cache, target_vec, network->work_delta, network->loss_type);

    // Calcul de la dérivée de l'activation pour la dernière couche (réutilise work_act_deriv)
    if (network->output_activation == ACTIVATION_SOFTMAX && network->loss_type == LOSS_CATEGORICAL_CROSS_ENTROPY) {
        // Forme simplifiee: dL/dz = y_pred - y_true.
        for (size_t i = 0; i < output_size; i++) {
            network->work_delta->data[i] =
                network->layers[network->num_layers - 1]->output_cache->data[i] - target_vec->data[i];
        }
    } else {
        nn_activation_derivative(network->layers[network->num_layers - 1]->output_cache,
                                 network->work_act_deriv,
                                 network->output_activation);

        // Multiplication élément par élément: delta = loss' * activation'
        for (size_t i = 0; i < output_size; i++) {
            network->work_delta->data[i] *= network->work_act_deriv->data[i];
        }
    }

    // Ping-pong entre work_delta et work_prev_delta pour éviter tout malloc dans la boucle
    Vector *delta      = network->work_delta;
    Vector *next_delta = network->work_prev_delta;

    // Rétropropagation à travers chaque couche
    for (size_t l = network->num_layers; l > 0; l--) {
        Layer *layer = network->layers[l - 1];
        Vector *current_delta = delta;

        // Propager l'erreur à la couche précédente (si ce n'est pas la première couche)
        if (l > 1) {
            size_t prev_size = layer->input_cache->size;

            // delta_{l-1} = W_l^T * delta_l (sans la colonne biais)
            for (size_t i = 0; i < prev_size; i++) {
                next_delta->data[i] = 0.0f;
                for (size_t j = 0; j < layer->weights->rows; j++) {
                    next_delta->data[i] += layer->weights->data[j * layer->weights->cols + i] * current_delta->data[j];
                }
            }

            // Multiplier par la dérivée de l'activation de la couche précédente (réutilise work_act_deriv)
            nn_activation_derivative(layer->input_cache, network->work_act_deriv, network->hidden_activation);
            for (size_t i = 0; i < prev_size; i++) {
                next_delta->data[i] *= network->work_act_deriv->data[i];
            }

            // Échange des pointeurs (ping-pong), aucun malloc/free
            Vector *tmp = delta;
            delta       = next_delta;
            next_delta  = tmp;
        }

        if (network->optimizer_type == OPTIMIZER_ADAM && network->adam_opts && network->opt_gradients) {
            Matrix *grad = network->opt_gradients[l - 1];
            network->adam_opts[l - 1]->learning_rate = learning_rate;

            for (size_t i = 0; i < layer->weights->rows; i++) {
                for (size_t j = 0; j < layer->weights->cols; j++) {
                    grad->data[i * layer->weights->cols + j] = current_delta->data[i] * layer->bias_input->data[j];
                }
            }

            nn_adam_update(network->adam_opts[l - 1], layer->weights, grad);
        } else {
            // Mise a jour SGD en ligne: W = W - lr * (delta * input^T).
            for (size_t i = 0; i < layer->weights->rows; i++) {
                for (size_t j = 0; j < layer->weights->cols; j++) {
                    const size_t idx = i * layer->weights->cols + j;
                    layer->weights->data[idx] -= learning_rate * current_delta->data[i] * layer->bias_input->data[j];
                }
            }
        }
    }

    // Aucun free nécessaire
}

/* Compatibilite API: aucun gradient persistant a remettre a zero. */
void nn_network_zero_gradients(Network *network) {
    (void)network;
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
        fprintf(stderr, "Echec de l'ouverture du fichier pour la sauvegarde : %s\n", filepath);
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
    printf("Reseau sauvegarde dans %s\n", filepath);
}

void nn_network_load(Network *network, const char *filepath) {
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "Echec de l'ouverture du fichier pour le chargement : %s\n", filepath);
        return;
    }
    
    // Charger et vérifier la structure
    size_t num_layers;
    fread(&num_layers, sizeof(size_t), 1, file);
    if (num_layers != network->num_layers) {
        fprintf(stderr, "Incompatibilité de structure du reseau\n");
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
    printf("Reseau charge depuis %s\n", filepath);
}

void nn_network_copy(Network *dest, const Network *src) {
    if (dest->num_layers != src->num_layers) {
        fprintf(stderr, "Impossible de copier des reseaux de structures differentes\n");
        return;
    }
    
    for (size_t i = 0; i < src->num_layers; i++) {
        copy_matrix(src->layers[i]->weights, dest->layers[i]->weights);
    }
}

/* Affichage du réseau */
void nn_network_print(const Network *network) {
    printf("Structure du reseau :\n");
    printf("  Couches : %zu\n", network->num_layers + 1);
    for (size_t i = 0; i <= network->num_layers; i++) {
        printf("    Couche %zu : %zu neurones\n", i, network->layer_sizes[i]);
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
        printf("Epoque %d/%d terminee\n", i+1, epochs);
    }
    free_vector(input);
    free_vector(target);
}
