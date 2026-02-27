#include "nn/network.h"
#include "nn/nn_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

static int float_eq(float a, float b, float epsilon) {
    return fabs(a - b) < epsilon;
}

int main(void) {
    printf("Tests du reseau de neurones...\n");

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

    /* Test de remise à zéro des gradients */
    {
        size_t layer_sizes[] = {2, 3, 2};
        Network *net = nn_network_create(layer_sizes, 3);

        for (size_t i = 0; i < net->num_layers; i++) {
            for (size_t j = 0; j < net->layers[i]->gradients->rows * net->layers[i]->gradients->cols; j++) {
                net->layers[i]->gradients->data[j] = 1.0f;
            }
        }

        nn_network_zero_gradients(net);

        for (size_t i = 0; i < net->num_layers; i++) {
            for (size_t j = 0; j < net->layers[i]->gradients->rows * net->layers[i]->gradients->cols; j++) {
                assert(float_eq(net->layers[i]->gradients->data[j], 0.0f, 0.001f));
            }
        }

        nn_network_free(net);
        printf("  [PASS] Remise a zero des gradients\n");
    }

    printf("Tous les tests du reseau sont passes !\n");
    return 0;
}
