#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include <stddef.h>
#include "matrix.h"
#include "nn_functions.h"
#include "mnist.h"

/* Structure d'une couche du réseau */
typedef struct {
    Matrix *weights;        // Poids W (biais inclus via l'astuce [W|1])
    Vector *input_cache;    // Entrée mise en cache pour la rétropropagation
    Vector *output_cache;   // Sortie mise en cache
    Vector *bias_input;     // Entrée concaténée avec le biais
} Layer;

/* Structure du réseau de neurones */
typedef struct {
    size_t num_layers;      // Nombre de couches de poids
    size_t *layer_sizes;    // Tailles de chaque couche
    Layer **layers;         // Tableau de pointeurs vers les couches

    // Configuration d'entrainement appliquee a toutes les couches du reseau.
    ActivationType hidden_activation;
    ActivationType output_activation;
    LossType loss_type;
    OptimizerType optimizer_type;

    // Etat interne des optimiseurs adaptatifs (alloue seulement si necessaire).
    AdamOptimizer **adam_opts;
    Matrix **opt_gradients;

    // Buffers de travail pre-alloues pour eviter les allocations dans la boucle d'entrainement.
    Vector *work_predicted;
    Vector *work_one_hot;
    Vector *work_delta;
    Vector *work_prev_delta;
    Vector *work_act_deriv;
} Network;

// Création et destruction
Network* nn_network_create(const size_t *layer_sizes, size_t num_layers);
void nn_network_free(Network *network);

// Propagation avant
void nn_network_forward(const Network *network, const Vector *input);

// Rétropropagation
void nn_network_backward(Network *network, const Vector *input, const Vector *target, float learning_rate);

// Configuration de l'entrainement
void nn_network_set_training_config(Network *network,
                                    ActivationType hidden_activation,
                                    ActivationType output_activation,
                                    LossType loss_type,
                                    OptimizerType optimizer_type);

// Prédiction
void nn_network_predict(const Network *network, const Vector *input, size_t *predicted_class);

// Sauvegarde et chargement
void nn_network_save(const Network *network, const char *filepath);
void nn_network_load(Network *network, const char *filepath);
void nn_network_copy(Network *dest, const Network *src);
void nn_network_print(const Network *network);

// Remise à zéro des gradients
void nn_network_zero_gradients(Network *network);

// Entrainement et évaluation
float nn_network_evaluate(const Network *network, const mnist_dataset_t *dataset);
void nn_network_train(Network *network, mnist_dataset_t *train_data, float learning_rate, int epochs);

#endif // NN_NETWORK_H