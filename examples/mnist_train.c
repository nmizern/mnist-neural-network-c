#include "neuralnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//ALL IS GENERATED NOW IT'S JUST FOR TEST LOADING MNIST DATASET, NOT TRAINING

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

    // 1. Load training dataset
    printf("Loading training data...\n");
    mnist_dataset_t *train = mnist_load_dataset(data_path, "train");
    if (!train) {
        fprintf(stderr, "Failed to load training data\n");
        return 1;
    }
    printf("Loaded %zu training samples\n", train->count);

    // 2. Load test dataset
    printf("Loading test data...\n");
    mnist_dataset_t *test = mnist_load_dataset(data_path, "test");
    if (!test) {
        fprintf(stderr, "Failed to load test data\n");
        mnist_free_dataset(train);
        return 1;
    }
    printf("Loaded %zu test samples\n", test->count);

    // 3. Print first image as ASCII art
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

    // 4. Test shuffle
    printf("\nTesting shuffle...\n");
    printf("Before shuffle: first label = %d\n", train->labels[0]);
    mnist_shuffle(train);
    printf("After shuffle: first label = %d\n", train->labels[0]);

    // 5. Cleanup
    mnist_free_dataset(train);
    mnist_free_dataset(test);

    printf("\nMNIST loading test passed!\n");

    return 0;
}
