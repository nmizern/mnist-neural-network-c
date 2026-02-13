#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include <stddef.h>
#include "matrix.h"
#include "nn_functions.h"
#include "mnist.h"

// Structure représentant une couche unique
// Nous avons besoin d'une structure dédiée, pas juste une matrice, car
// l'algorithme de Backpropagation nécessite de stocker les poids, gradients, et d'autres quantités
typedef struct {
    Matrix *weights;        // Matrice des poids W (incluant les biais via l'astuce [W|1])
    Matrix *gradients;
    
    // Valeurs mises en cache pour la Backpropagation
    Vector *input_cache;
    Vector *output_cache;
} Layer;
// Je pensais qu'une structure de tensor pour le reseau directe serait plus simple, comme nous
// l'avons discuté avec le professeur, mais pour la backpropagation on a du rajouter des vecteurs de cache

typedef struct {
    size_t num_layers;      // Nombre de couches
    size_t *layer_sizes;    // Tableau des tailles de couches
    Layer **layers;        // Tableau de pointeurs vers les couches
} Network;

Network* nn_network_create(const size_t *layer_sizes, size_t num_layers);
void nn_network_free(Network *network);

// Propagation Avant (for Training and Inference)
void nn_network_forward(const Network *network, const Vector *input, Vector *output);

// Rétropropagation (Training)
// Calcule les gradients en remontant de la fin vers le début
void nn_network_backward(Network *network, const Vector *input, const Vector *target, float learning_rate);

// Prédiction simple (Inference)
void nn_network_predict(const Network *network, const Vector *input, size_t *predicted_class);

void nn_network_save(const Network *network, const char *filepath);
void nn_network_load(Network *network, const char *filepath);
void nn_network_copy(Network *dest, const Network *src);
void nn_network_print(const Network *network);

// Remise à zéro des gradients avant une nouvelle passe
void nn_network_zero_gradients(Network *network);

float nn_network_evaluate(const Network *network, const mnist_dataset_t *dataset);

#endif // NN_NETWORK_H