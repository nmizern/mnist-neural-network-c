#include "neuralnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

Network mnist_nn() {
    size_t layer_sizes[] = {MNIST_IMAGE_SIZE, 128, 64, MNIST_NUM_CLASSES};
    size_t num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
    return *nn_network_create(layer_sizes, num_layers);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    printf("MNIST Neural Network Training\n");
    printf("==============================\n\n");

    Network network = mnist_nn();

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

// LOAD AND SPLIT DATASET

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

    printf("\nFirst training image (label=%d):\n", train->labels[0]);
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            float pixel = train->images[0 * 784 + y * 28 + x];
            if (pixel > 0.5f) printf("##");
            else if (pixel > 0.2f) printf("..");
            else printf("  ");
        }
        printf("\n");
    }

    
    printf("\nTesting shuffle...\n");
    printf("Before shuffle: first label = %d\n", train->labels[0]);
    mnist_shuffle(train);
    printf("After shuffle: first label = %d\n", train->labels[0]);

    nn_network_train(&network, train, 0.01f, 1);
    nn_network_print(&network);
    float accuracy = nn_network_evaluate(&network, test);
    printf("\nTest accuracy after 1 epoch: %.2f%%\n", accuracy * 100.0f);

    mnist_free_dataset(train);
    mnist_free_dataset(test);

    return 0;
}
