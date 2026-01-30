/**
 * @file network.h
 * @brief Network structure, inference, training, and evaluation
 *
 * This is the central module. It covers all 5 requirements from the PDF:
 *
 * Req 2: Create network with variable layers/sizes
 *   network_create([784, 128, 10], 3) -> network with 2 layers
 *
 * Req 3: Inference (PDF eq. 5)
 *   y_hat = argmax[ sigma(W_2^T * sigma(W_1^T * sigma(W_0^T * x + B_0) + B_1) + B_2) ]
 *
 * Req 4: Measure quality
 *   accuracy = correct_predictions / total_predictions on test set
 *
 * Req 5: Training (PDF Algorithm 1 - SGD)
 *   For each sample: forward -> loss -> backpropagation -> update weights
 *   W_k = W_k - eta * dl/dW_k
 *   B_k = B_k - eta * dl/dB_k
 *
 * Req 1: Gradient computation (dl/dW, dl/dB) is implemented
 *   inside the backpropagation part of training.
 */

#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include <stddef.h>
#include "matrix.h"

/* TODO: Network structure (layers with W, B, stored activations) */
/* TODO: network_create(sizes[], num_sizes) */
/* TODO: network_destroy(net) */
/* TODO: network_forward(net, input) - inference */
/* TODO: network_predict(net, input) - returns class 0-9 (argmax) */
/* TODO: network_train(net, dataset, learning_rate, epochs) - SGD training */
/* TODO: network_accuracy(net, dataset) - evaluate on test set */

#endif /* NN_NETWORK_H */
