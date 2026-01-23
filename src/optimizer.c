/**
 * @file optimizer.c
 * @brief Training optimization implementation
 *
 * Implement the functions declared in nn/optimizer.h here.
 *
 * THIS IS THE MOST IMPORTANT FILE FOR THE PROJECT.
 * Understanding and implementing backpropagation is key.
 *
 * BACKPROPAGATION ALGORITHM:
 *
 * Given: network output x^(K), true label y, loss gradient dl/dx^(K)
 *
 * For k = K-1 down to 0:
 *   1. Get stored values from forward pass:
 *      - z^(k) = pre-activation (before applying activation)
 *      - x^(k) = input to this layer
 *
 *   2. Compute gradient w.r.t. pre-activation:
 *      dl/dz^(k) = dl/dx^(k+1) * sigma'(z^(k))  [element-wise]
 *
 *   3. Compute weight gradient:
 *      dl/dW_k = x^(k) * (dl/dz^(k))^T
 *      This is an outer product: result is (input_size x output_size)
 *
 *   4. Compute bias gradient:
 *      dl/dB_k = dl/dz^(k)
 *
 *   5. Propagate gradient to previous layer:
 *      dl/dx^(k) = W_k * dl/dz^(k)
 *
 * SGD UPDATE:
 *   W_k = W_k - learning_rate * dl/dW_k
 *   B_k = B_k - learning_rate * dl/dB_k
 *
 * TRAINING LOOP:
 *   for epoch in 1..num_epochs:
 *       for (image, label) in training_data:
 *           1. output = forward(network, image)
 *           2. loss = mse_loss(output, label)
 *           3. grad = mse_gradient(output, label)
 *           4. backpropagate(network, grad)
 *           5. sgd_update(network, learning_rate)
 *       print("Epoch", epoch, "complete")
 *       accuracy = evaluate(network, test_data)
 *       print("Test accuracy:", accuracy)
 */

#include "nn/optimizer.h"
#include "nn/loss.h"
#include <stdlib.h>
#include <stdio.h>

/* TODO: Implement backpropagation */

/* TODO: Implement SGD update */

/* TODO: Implement training loop */
