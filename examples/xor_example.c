/**
 * @file xor_example.c
 * @brief XOR problem - simple test for neural network
 *
 * The XOR problem is a classic test for neural networks.
 * It requires at least one hidden layer because XOR is not linearly separable.
 *
 * XOR Truth Table:
 *   Input    | Output
 *   0, 0     | 0
 *   0, 1     | 1
 *   1, 0     | 1
 *   1, 1     | 0
 *
 * RECOMMENDED NETWORK:
 * - Input: 2 neurons
 * - Hidden: 4-8 neurons with Sigmoid or ReLU
 * - Output: 1 neuron with Sigmoid
 *
 * This is a good first test before tackling MNIST:
 * 1. Create network with sizes [2, 4, 1]
 * 2. Train on 4 XOR samples
 * 3. Test predictions
 *
 * If XOR works, your forward/backward propagation is likely correct.
 */

#include "neuralnet.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("XOR Neural Network Example\n");
    printf("==========================\n\n");

    /* TODO: Implement XOR training example
     *
     * Steps:
     * 1. Define XOR training data
     *    float inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
     *    float targets[4] = {0, 1, 1, 0};
     *
     * 2. Create network: [2, 4, 1]
     *
     * 3. Train for ~10000 epochs with learning_rate ~0.5
     *
     * 4. Test predictions and print results
     */

    printf("TODO: Implement XOR example\n");

    return 0;
}
