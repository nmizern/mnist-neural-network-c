#include "nn/network.h"
#include "nn/nn_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>

static int float_eq(float a, float b, float epsilon) {
    return fabs(a - b) < epsilon;
}

static size_t eval_or_accuracy(Network *net) {
    const float samples[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    const size_t labels[4] = {0, 1, 1, 1};
    size_t correct = 0;

    Vector *input = create_vector(2);
    for (size_t i = 0; i < 4; i++) {
        input->data[0] = samples[i][0];
        input->data[1] = samples[i][1];
        size_t pred = 0;
        nn_network_predict(net, input, &pred);
        if (pred == labels[i]) {
            correct++;
        }
    }
    free_vector(input);
    return correct;
}

static void train_on_or(Network *net, int epochs, float learning_rate) {
    const float samples[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    const size_t labels[4] = {0, 1, 1, 1};

    Vector *input = create_vector(2);
    Vector *target = create_vector(1);

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (size_t i = 0; i < 4; i++) {
            input->data[0] = samples[i][0];
            input->data[1] = samples[i][1];
            target->data[0] = (float)labels[i];
            nn_network_backward(net, input, target, learning_rate);
        }
    }

    free_vector(input);
    free_vector(target);
}

int main(void) {
    printf("Tests du reseau de neurones...\n");
    srand((unsigned int)time(NULL));

    /* Test de création du réseau */
    {
        size_t layer_sizes[] = {3, 5, 2};
        Network *net = nn_network_create(layer_sizes, 3);

        assert(net != NULL);
        assert(net->num_layers == 2);
        assert(net->layer_sizes[0] == 3);
        assert(net->layer_sizes[1] == 5);
        assert(net->layer_sizes[2] == 2);

        nn_network_free(net);
        printf("  [PASS] Creation du reseau\n");
    }

    /* Test de prédiction */
    {
        size_t layer_sizes[] = {2, 4, 3};
        Network *net = nn_network_create(layer_sizes, 3);

        Vector *input = create_vector(2);
        input->data[0] = 1.0f;
        input->data[1] = 0.5f;

        size_t predicted_class;
        nn_network_predict(net, input, &predicted_class);

        assert(predicted_class < 3);

        free_vector(input);
        nn_network_free(net);
        printf("  [PASS] Prediction\n");
    }

    /* Test de compatibilite de nn_network_zero_gradients (no-op) */
    {
        size_t layer_sizes[] = {2, 3, 2};
        Network *net = nn_network_create(layer_sizes, 3);

        float before = net->layers[0]->weights->data[0];

        nn_network_zero_gradients(net);

        assert(float_eq(net->layers[0]->weights->data[0], before, 0.001f));

        nn_network_free(net);
        printf("  [PASS] Zero gradients (no-op)\n");
    }

    /* Test d'entrainement rapide sur OR pour plusieurs architectures */
    {
        const size_t a1[] = {2, 2, 2};
        const size_t a2[] = {2, 4, 2};
        const size_t a3[] = {2, 6, 2};

        const size_t *architectures[] = {a1, a2, a3};

        for (size_t i = 0; i < 3; i++) {
            Network *net = nn_network_create(architectures[i], 3);
            size_t before = eval_or_accuracy(net);

            train_on_or(net, 120, 0.2f);

            size_t after = eval_or_accuracy(net);
            printf("    Architecture %zu-%zu-%zu : avant=%zu/4 apres=%zu/4\n",
                   architectures[i][0], architectures[i][1], architectures[i][2], before, after);
            assert(after >= before);
            assert(after >= 3);

            nn_network_free(net);
        }
        printf("  [PASS] Entrainement OR (architectures simples)\n");
    }

    printf("Tous les tests du reseau sont passes !\n");
    return 0;
}
