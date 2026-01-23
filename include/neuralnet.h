/**
 * @file neuralnet.h
 * @brief Main public API header - includes all neural network modules
 *
 * This is the main header file that users of the library should include.
 * It provides access to all neural network functionality.
 *
 * USAGE:
 *   #include "neuralnet.h"
 *
 * This will include all submodules:
 * - matrix.h    : Matrix/vector operations
 * - activation.h: Activation functions (ReLU, Sigmoid)
 * - layer.h     : Single layer structure
 * - network.h   : Complete network structure
 * - loss.h      : Loss functions (MSE)
 * - optimizer.h : Training algorithms (SGD)
 * - mnist.h     : MNIST dataset loading
 * - metrics.h   : Evaluation metrics (accuracy)
 */

#ifndef NEURALNET_H
#define NEURALNET_H

#include "nn/matrix.h"
#include "nn/activation.h"
#include "nn/layer.h"
#include "nn/network.h"
#include "nn/loss.h"
#include "nn/optimizer.h"
#include "nn/mnist.h"
#include "nn/metrics.h"

#endif /* NEURALNET_H */
