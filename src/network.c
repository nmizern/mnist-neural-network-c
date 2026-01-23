/**
 * @file network.c
 * @brief Network operations implementation
 *
 * Implement the functions declared in nn/network.h here.
 *
 * IMPLEMENTATION NOTES:
 *
 * Network creation from sizes array:
 *   sizes = [784, 128, 64, 10]
 *   Creates 3 layers:
 *     Layer 0: 784 -> 128 (with activation)
 *     Layer 1: 128 -> 64  (with activation)
 *     Layer 2: 64 -> 10   (output layer)
 *
 *   Number of layers = len(sizes) - 1
 *
 * Forward propagation:
 *   input = x
 *   for each layer:
 *       input = layer_forward(layer, input)
 *   return input (final output)
 *
 * Prediction:
 *   output = forward(network, input)
 *   return argmax(output)  // index of maximum value
 *
 * ACTIVATION CHOICE:
 * - Hidden layers: ReLU (faster training) or Sigmoid
 * - Output layer: Often no activation or Softmax for classification
 *   With MSE loss, you can use Sigmoid on output layer
 */

#include "nn/network.h"
#include <stdlib.h>
#include <stdio.h>

/* TODO: Implement network functions */
