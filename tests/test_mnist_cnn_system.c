#include "nn/network.h"
#include "nn/mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_TRAIN_BIN "../../../data/train.bin"
#define DEFAULT_TEST_BIN  "../../../data/test.bin"
#define CNN_INPUT_SIZE    (14 * 14)

static size_t min_size(size_t a, size_t b) {
    return a < b ? a : b;
}

static void avgpool2x2_28_to_14(const float *src, float *dst) {
    for (size_t y = 0; y < 14; y++) {
        for (size_t x = 0; x < 14; x++) {
            size_t i0 = (2 * y) * 28 + (2 * x);
            size_t i1 = i0 + 1;
            size_t i2 = i0 + 28;
            size_t i3 = i2 + 1;
            dst[y * 14 + x] = 0.25f * (src[i0] + src[i1] + src[i2] + src[i3]);
        }
    }
}

static float eval_model(Network *net, const mnist_dataset_t *dataset, size_t count, int use_pooling) {
    size_t n = min_size(count, dataset->count);
    size_t input_size = use_pooling ? CNN_INPUT_SIZE : MNIST_IMAGE_SIZE;
    Vector *input = create_vector(input_size);
    float pooled[CNN_INPUT_SIZE];

    size_t correct = 0;
    for (size_t i = 0; i < n; i++) {
        const float *img = &dataset->images[i * MNIST_IMAGE_SIZE];

        if (use_pooling) {
            avgpool2x2_28_to_14(img, pooled);
            for (size_t j = 0; j < CNN_INPUT_SIZE; j++) {
                input->data[j] = pooled[j];
            }
        } else {
            for (size_t j = 0; j < MNIST_IMAGE_SIZE; j++) {
                input->data[j] = img[j];
            }
        }

        size_t pred = 0;
        nn_network_predict(net, input, &pred);
        if (pred == dataset->labels[i]) {
            correct++;
        }
    }

    free_vector(input);
    return n == 0 ? 0.0f : (float)correct / (float)n;
}

static void train_model(Network *net, mnist_dataset_t *train, size_t train_count, float lr, int epochs, int use_pooling) {
    size_t n = min_size(train_count, train->count);
    size_t input_size = use_pooling ? CNN_INPUT_SIZE : MNIST_IMAGE_SIZE;
    Vector *input = create_vector(input_size);
    Vector *label = create_vector(1);
    float pooled[CNN_INPUT_SIZE];

    for (int e = 0; e < epochs; e++) {
        mnist_shuffle(train);
        for (size_t i = 0; i < n; i++) {
            const float *img = &train->images[i * MNIST_IMAGE_SIZE];

            if (use_pooling) {
                avgpool2x2_28_to_14(img, pooled);
                for (size_t j = 0; j < CNN_INPUT_SIZE; j++) {
                    input->data[j] = pooled[j];
                }
            } else {
                for (size_t j = 0; j < MNIST_IMAGE_SIZE; j++) {
                    input->data[j] = img[j];
                }
            }

            label->data[0] = (float)train->labels[i];
            nn_network_backward(net, input, label, lr);
        }
    }

    free_vector(input);
    free_vector(label);
}

int main(int argc, char *argv[]) {
    const char *train_path = (argc > 1) ? argv[1] : DEFAULT_TRAIN_BIN;
    const char *test_path = (argc > 2) ? argv[2] : DEFAULT_TEST_BIN;

    printf("Benchmark systeme CNN-style MNIST\n");
    printf("=================================\n");
    printf("Comparaison : MLP classique vs MLP avec entree poolee (style CNN).\n");

    mnist_dataset_t *train = mnist_load_binary(train_path);
    mnist_dataset_t *test = mnist_load_binary(test_path);
    if (!train || !test) {
        fprintf(stderr, "Echec du chargement des binaires MNIST.\n");
        fprintf(stderr, "Usage : test_mnist_cnn_system [train.bin] [test.bin]\n");
        mnist_free_dataset(train);
        mnist_free_dataset(test);
        return 1;
    }

    const size_t train_subset = 2500;
    const size_t test_subset = 800;
    const int epochs = 3;

    printf("Sous-ensemble : train=%zu, test=%zu, epoques=%d\n",
           min_size(train_subset, train->count),
           min_size(test_subset, test->count),
           epochs);

    size_t dense_layers[] = {784, 128, 10};
    size_t cnn_style_layers[] = {CNN_INPUT_SIZE, 64, 10};

    Network *dense = nn_network_create(dense_layers, 3);
    Network *cnn_style = nn_network_create(cnn_style_layers, 3);

    float dense_before = eval_model(dense, test, test_subset, 0);
    float cnn_before = eval_model(cnn_style, test, test_subset, 1);

    printf("\nAvant entrainement\n");
    printf("  Dense classique     : %6.2f%%\n", dense_before * 100.0f);
    printf("  Style CNN (pool 2x2): %6.2f%%\n", cnn_before * 100.0f);

    train_model(dense, train, train_subset, 0.08f, epochs, 0);
    train_model(cnn_style, train, train_subset, 0.10f, epochs, 1);

    float dense_after = eval_model(dense, test, test_subset, 0);
    float cnn_after = eval_model(cnn_style, test, test_subset, 1);

    printf("\nApres entrainement\n");
    printf("  Dense classique     : %6.2f%% (delta %+6.2f%%)\n",
           dense_after * 100.0f,
           (dense_after - dense_before) * 100.0f);
    printf("  Style CNN (pool 2x2): %6.2f%% (delta %+6.2f%%)\n",
           cnn_after * 100.0f,
           (cnn_after - cnn_before) * 100.0f);

    printf("\nInterpretation : le modele a entree poolee est un systeme leger de type CNN pour une comparaison structurelle rapide.\n");

    nn_network_free(dense);
    nn_network_free(cnn_style);
    mnist_free_dataset(train);
    mnist_free_dataset(test);
    return 0;
}
