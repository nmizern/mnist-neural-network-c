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
    printf("Running activation tests...\n");

    /* Test sigmoid */
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
        printf("  [PASS] sigmoid\n");
    }

    /* Test ReLU */
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
    
    /* Test softmax */
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
    
    /* Test sigmoid derivative */
    {
        Vector *output = create_vector(2);
        Vector *result = create_vector(2);
        output->data[0] = 0.5f;
        output->data[1] = 0.731f;
        
        nn_sigmoid_derivative(output, result);
        
        assert(float_eq(result->data[0], 0.25f, 0.001f));
        
        free_vector(output);
        free_vector(result);
        printf("  [PASS] Sigmoid derivative\n");
    }
    
    /* Test MSE loss */
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
        printf("  [PASS] MSE loss\n");
    }
    
    /* Test argmax */
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

    printf("All activation tests passed!\n");
    return 0;
}
