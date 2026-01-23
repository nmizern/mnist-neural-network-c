/**
 * @file metrics.h
 * @brief Evaluation metrics for classification quality
 *
 * This module implements functions to measure the quality of network
 * predictions, as required by project PDF section 5, point 4.
 *
 * REQUIRED FUNCTIONALITY:
 *
 * 1. Accuracy calculation:
 *    accuracy = correct_predictions / total_predictions
 *
 *    Evaluate on test set to measure generalization.
 *
 * 2. Per-class accuracy (optional but useful):
 *    How well does the network recognize each digit?
 *
 * USAGE:
 * After training, run inference on the test dataset and compute accuracy.
 * This tells you how well the network generalizes to unseen data.
 *
 * TYPICAL MNIST RESULTS:
 * - Random guessing: ~10% accuracy
 * - Simple 2-layer network: 95-97% accuracy
 * - Well-tuned network: 98%+ accuracy
 *
 * FOR FUTURE (scientific paper):
 * - Confusion matrix
 * - Precision, recall, F1-score per class
 * - Training/validation loss curves
 */

#ifndef NN_METRICS_H
#define NN_METRICS_H

#include <stddef.h>
#include "network.h"
#include "mnist.h"

/* TODO: Calculate accuracy on dataset */
/* Input: network, dataset (images + labels) */
/* Output: accuracy as float [0, 1] */

/* TODO: Evaluate network and print results */
/* Runs inference on all test samples, prints accuracy */

#endif /* NN_METRICS_H */
