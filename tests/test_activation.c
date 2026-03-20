#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "nn/nn_functions.h"
#include "nn/matrix.h"

static int float_eq(float a, float b, float epsilon) {
    return fabs(a - b) < epsilon;
}

int main(void) {
    printf("Tests des fonctions d'activation...\n");

    /* Sigmoide */
    {
        Vector *input = create_vector(3);
        Vector *output = create_vector(3);
        input->data[0] = 0.0f;
        input->data[1] = 1.0f;
        input->data[2] = -1.0f;

        nn_sigmoid(input, output);

        assert(float_eq(output->data[0], 0.5f, 0.001f));
        assert(float_eq(output->data[1], 0.731f, 0.01f));
        assert(float_eq(output->data[2], 0.269f, 0.01f));

        free_vector(input);
        free_vector(output);
        printf("  [PASS] Sigmoide\n");
    }

    /* ReLU */
    {
        Vector *input = create_vector(4);
        Vector *output = create_vector(4);
        input->data[0] = 5.0f;
        input->data[1] = -3.0f;
        input->data[2] = 0.0f;
        input->data[3] = 2.5f;

        nn_relu(input, output);

        assert(float_eq(output->data[0], 5.0f, 0.001f));
        assert(float_eq(output->data[1], 0.0f, 0.001f));
        assert(float_eq(output->data[2], 0.0f, 0.001f));
        assert(float_eq(output->data[3], 2.5f, 0.001f));

        free_vector(input);
        free_vector(output);
        printf("  [PASS] ReLU\n");
    }

    /* Softmax */
    {
        Vector *input = create_vector(3);
        Vector *output = create_vector(3);
        input->data[0] = 1.0f;
        input->data[1] = 2.0f;
        input->data[2] = 3.0f;

        nn_softmax(input, output);

        float sum = output->data[0] + output->data[1] + output->data[2];
        assert(float_eq(sum, 1.0f, 0.001f));
        assert(output->data[2] > output->data[1]);
        assert(output->data[1] > output->data[0]);

        free_vector(input);
        free_vector(output);
        printf("  [PASS] Softmax\n");
    }

    /* Tanh */
    {
        Vector *input = create_vector(3);
        Vector *output = create_vector(3);
        input->data[0] = 0.0f;
        input->data[1] = 1.0f;
        input->data[2] = -1.0f;

        nn_tanh(input, output);

        assert(float_eq(output->data[0], 0.0f, 0.001f));
        assert(float_eq(output->data[1], 0.761f, 0.01f));
        assert(float_eq(output->data[2], -0.761f, 0.01f));

        free_vector(input);
        free_vector(output);
        printf("  [PASS] Tanh\n");
    }

    /* Leaky ReLU */
    {
        Vector *input = create_vector(3);
        Vector *output = create_vector(3);
        input->data[0] = -2.0f;
        input->data[1] = 0.0f;
        input->data[2] = 3.0f;

        nn_leaky_relu(input, output, 0.1f);

        assert(float_eq(output->data[0], -0.2f, 0.001f));
        assert(float_eq(output->data[1], 0.0f, 0.001f));
        assert(float_eq(output->data[2], 3.0f, 0.001f));

        free_vector(input);
        free_vector(output);
        printf("  [PASS] Leaky ReLU\n");
    }

    /* Dérivée de la sigmoide */
    {
        Vector *output = create_vector(2);
        Vector *result = create_vector(2);
        output->data[0] = 0.5f;
        output->data[1] = 0.731f;

        nn_sigmoid_derivative(output, result);

        assert(float_eq(result->data[0], 0.25f, 0.001f));

        free_vector(output);
        free_vector(result);
        printf("  [PASS] Derivee de la sigmoide\n");
    }

    /* Perte MSE */
    {
        Vector *predicted = create_vector(3);
        Vector *target = create_vector(3);
        predicted->data[0] = 1.0f;
        predicted->data[1] = 2.0f;
        predicted->data[2] = 3.0f;
        target->data[0] = 1.5f;
        target->data[1] = 2.5f;
        target->data[2] = 2.5f;

        float loss = nn_loss_mse(predicted, target);
        assert(float_eq(loss, 0.25f, 0.01f));

        free_vector(predicted);
        free_vector(target);
        printf("  [PASS] Perte MSE\n");
    }

    /* Perte MAE */
    {
        Vector *predicted = create_vector(3);
        Vector *target = create_vector(3);
        predicted->data[0] = 1.0f;
        predicted->data[1] = 2.0f;
        predicted->data[2] = 3.0f;
        target->data[0] = 1.5f;
        target->data[1] = 2.5f;
        target->data[2] = 2.5f;

        float loss = nn_loss_mae(predicted, target);
        assert(float_eq(loss, 0.5f, 0.01f));

        free_vector(predicted);
        free_vector(target);
        printf("  [PASS] Perte MAE\n");
    }

    /* BCE + gradient */
    {
        Vector *predicted = create_vector(2);
        Vector *target = create_vector(2);
        Vector *grad = create_vector(2);

        predicted->data[0] = 0.9f;
        predicted->data[1] = 0.2f;
        target->data[0] = 1.0f;
        target->data[1] = 0.0f;

        float loss = nn_loss_binary_cross_entropy(predicted, target);
        assert(loss > 0.0f);

        nn_loss_binary_cross_entropy_gradient(predicted, target, grad);
        assert(grad->data[0] < 0.0f);
        assert(grad->data[1] > 0.0f);

        free_vector(predicted);
        free_vector(target);
        free_vector(grad);
        printf("  [PASS] BCE\n");
    }

    /* CCE */
    {
        Vector *predicted = create_vector(3);
        Vector *target = create_vector(3);
        predicted->data[0] = 0.1f;
        predicted->data[1] = 0.8f;
        predicted->data[2] = 0.1f;
        target->data[0] = 0.0f;
        target->data[1] = 1.0f;
        target->data[2] = 0.0f;

        float loss = nn_loss_categorical_cross_entropy(predicted, target);
        assert(float_eq(loss, -logf(0.8f) / 3.0f, 0.01f));

        free_vector(predicted);
        free_vector(target);
        printf("  [PASS] CCE\n");
    }

    /* API générique des pertes */
    {
        Vector *predicted = create_vector(2);
        Vector *target = create_vector(2);
        Vector *grad = create_vector(2);
        predicted->data[0] = 0.7f;
        predicted->data[1] = 0.3f;
        target->data[0] = 1.0f;
        target->data[1] = 0.0f;

        float loss = nn_loss_compute(predicted, target, LOSS_MAE);
        assert(loss > 0.0f);

        nn_loss_gradient(predicted, target, grad, LOSS_MAE);
        assert(grad->data[0] <= 0.0f);

        free_vector(predicted);
        free_vector(target);
        free_vector(grad);
        printf("  [PASS] API generique des pertes\n");
    }

    /* Optimiseur Adam */
    {
        Matrix *weights = create_matrix(1, 2);
        Matrix *grads = create_matrix(1, 2);

        weights->data[0] = 1.0f;
        weights->data[1] = -1.0f;
        grads->data[0] = 0.5f;
        grads->data[1] = -0.5f;

        AdamOptimizer *adam = nn_adam_create(1, 2, 0.01f, 0.9f, 0.999f, 1e-8f);
        nn_adam_update(adam, weights, grads);

        assert(weights->data[0] < 1.0f);
        assert(weights->data[1] > -1.0f);

        nn_adam_free(adam);
        free_matrix(weights);
        free_matrix(grads);
        printf("  [PASS] Adam\n");
    }

    /* Argmax */
    {
        Vector *vec = create_vector(4);
        vec->data[0] = 0.1f;
        vec->data[1] = 0.5f;
        vec->data[2] = 0.3f;
        vec->data[3] = 0.2f;

        size_t max_idx = nn_argmax(vec);
        assert(max_idx == 1);

        free_vector(vec);
        printf("  [PASS] Argmax\n");
    }

    printf("Tous les tests d'activation sont passes !\n");
    return 0;
}
