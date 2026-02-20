#include "neuralnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    srand(time(NULL));
    printf("MNIST Neural Network Training\n");
    printf("==============================\n\n");

    if (argc < 2) {
        printf("Usage: %s <path_to_mnist_pngs>\n", argv[0]);
        printf("\nThe directory should contain:\n");
        printf("  train/0/, train/1/, ..., train/9/\n");
        printf("  test/0/, test/1/, ..., test/9/\n");
        printf("\nDataset: https://github.com/rasbt/mnist-pngs\n");
        return 1;
    }

    const char *data_path = argv[1];
    printf("Data path: %s\n\n", data_path);

    srand((unsigned int)time(NULL));

    printf("Loading training data...\n");
    mnist_dataset_t *train = mnist_load_dataset(data_path, "train");
    if (!train) {
        fprintf(stderr, "Failed to load training data\n");
        return 1;
    }
    printf("Loaded %zu training samples\n", train->count);

    printf("Loading test data...\n");
    mnist_dataset_t *test = mnist_load_dataset(data_path, "test");
    if (!test) {
        fprintf(stderr, "Failed to load test data\n");
        mnist_free_dataset(train);
        return 1;
    }
    printf("Loaded %zu test samples\n", test->count);

    const size_t layer_sizes[3] = {784, 128, 10};
    Network *network = nn_network_create(layer_sizes, 3);
    nn_network_print(network);

    nn_network_train(network, train, 0.1f, 3);

    float accuracy = nn_network_evaluate(network, test);
    printf("Test accuracy: %.2f%%\n", accuracy * 100.0f);

    nn_network_free(network);
    mnist_free_dataset(train);
    mnist_free_dataset(test);

    return 0;
}
