#include "neuralnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static mnist_dataset_t *load_split(const char *root, const char *split) {
    // Charge le cache binaire si disponible, sinon reconstruit le split depuis les PNG.
    char cache_path[512];
    snprintf(cache_path, sizeof(cache_path), "%s/%s.bin", root, split);

    mnist_dataset_t *dataset = mnist_load_binary(cache_path);
    if (dataset) {
        printf("Cache %s charge: %zu echantillons\n", split, dataset->count);
        return dataset;
    }

    dataset = mnist_load_dataset(root, split);
    if (!dataset) {
        fprintf(stderr, "Impossible de charger le jeu %s\n", split);
        return NULL;
    }

    mnist_save_binary(dataset, cache_path);
    printf("Jeu %s construit et mis en cache: %zu echantillons\n", split, dataset->count);
    return dataset;
}

int main(int argc, char **argv) {
    srand((unsigned int)time(NULL));

    if (argc < 2) {
        printf("Usage: %s <racine_mnist_png>\n", argv[0]);
        return 1;
    }

    mnist_dataset_t *train = load_split(argv[1], "train");
    if (!train) return 1;

    mnist_dataset_t *test = load_split(argv[1], "test");
    if (!test) {
        mnist_free_dataset(train);
        return 1;
    }

    const size_t layers[] = {784, 256, 128, 10};
    Network *network = nn_network_create(layers, 4);
    // Configuration volontairement classique pour servir de baseline.
    nn_network_set_training_config(network,
                                   ACTIVATION_SIGMOID,
                                   ACTIVATION_SIGMOID,
                                   LOSS_MSE,
                                   OPTIMIZER_SGD);

    printf("Config entrainement: hidden=sigmoid, output=sigmoid, loss=mse, optimizer=sgd\n");
    nn_network_train(network, train, 0.10f, 10);

    float accuracy = nn_network_evaluate(network, test);
    printf("Precision test: %.2f%%\n", accuracy * 100.0f);

    nn_network_save(network, "mnist_sigmoid_sgd_mse.bin");

    nn_network_free(network);
    mnist_free_dataset(train);
    mnist_free_dataset(test);
    return 0;
}
