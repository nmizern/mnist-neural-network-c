#include "neuralnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    printf("Entrainement du reseau de neurones MNIST\n");
    printf("=========================================\n\n");

    if (argc < 2) {
        printf("Usage : %s <chemin_vers_donnees_mnist>\n", argv[0]);
        printf("\nLe repertoire doit contenir :\n");
        printf("  train/0/, train/1/, ..., train/9/\n");
        printf("  test/0/, test/1/, ..., test/9/\n");
        printf("\nDataset : https://github.com/rasbt/mnist-pngs\n");
        return 1;
    }

    const char *data_path = argv[1];
    printf("Chemin des donnees : %s\n\n", data_path);

    // Essaie d'abord le cache binaire d'entrainement pour accelerer le lancement.
    char train_bin[512];
    snprintf(train_bin, sizeof(train_bin), "%s/train.bin", data_path);

    printf("Chargement des donnees d'entrainement...\n");
    mnist_dataset_t *train = mnist_load_binary(train_bin);
    if (train) {
        printf("Charge %zu echantillons d'entrainement depuis le cache\n", train->count);
    }else {
        train = mnist_load_dataset(data_path, "train");
        if (!train) {
            fprintf(stderr, "Echec du chargement des donnees d'entrainement\n");
            return 1;
        }
        printf("Charge %zu echantillons d'entrainement\n", train->count);
        mnist_save_binary(train, train_bin);
        printf("Cache d'entrainement sauvegarde dans %s\n", train_bin);
   }

    // Meme logique pour le split de test afin d'eviter de reparcourir les PNG.
    char test_bin[512];
    snprintf(test_bin, sizeof(test_bin), "%s/test.bin", data_path);

    printf("Chargement des donnees de test...\n");
    mnist_dataset_t *test = mnist_load_binary(test_bin);
    if (test) {
       printf("Charge %zu echantillons de test depuis le cache\n", test->count);
    } else {
       test = mnist_load_dataset(data_path, "test");
       if (!test) {
           fprintf(stderr, "Echec du chargement des donnees de test\n");
           mnist_free_dataset(train);
           return 1;
     }
        printf("Charge %zu echantillons de test\n", test->count);
        mnist_save_binary(test, test_bin);
        printf("Cache de test sauvegarde dans %s\n", test_bin);
    }

    const size_t layer_sizes[4] = {784, 256, 128, 10};
    // Architecture de base pour une premiere experience MNIST.
    Network *network = nn_network_create(layer_sizes, 4);
    nn_network_print(network);

    nn_network_train(network, train, 0.1f, 10);

    char model_bin[512];
    snprintf(model_bin, sizeof(model_bin), "%s/mnist_model.bin", data_path);
    nn_network_save(network, model_bin);

    float accuracy = nn_network_evaluate(network, test);
    printf("Precision sur le test : %.2f%%\n", accuracy * 100.0f);

    // Precision par chiffre
    size_t correct[10] = {0};
    size_t total[10] = {0};
    Vector *input = create_vector(784);
    for (size_t i = 0; i < test->count; i++) {
        for (size_t j = 0; j < 784; j++) {
            input->data[j] = test->images[i * 784 + j];
        }
        size_t predicted;
        nn_network_predict(network, input, &predicted);
        uint8_t label = test->labels[i];
        total[label]++;
        if (predicted == label) correct[label]++;
    }
    free_vector(input);
    printf("\nPrecision par chiffre :\n");
    for (int d = 0; d < 10; d++) {
        printf("  %d : %.2f%% (%zu/%zu)\n", d, 100.0f * correct[d] / total[d], correct[d], total[d]);
    }

    nn_network_free(network);
    mnist_free_dataset(train);
    mnist_free_dataset(test);

    return 0;
}
