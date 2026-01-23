/**
 * @file metrics.c
 * @brief Evaluation metrics implementation
 *
 * Implement the functions declared in nn/metrics.h here.
 *
 * ACCURACY CALCULATION:
 *
 * int correct = 0;
 * for each (image, label) in dataset:
 *     prediction = network_predict(network, image)
 *     if (prediction == label):
 *         correct++
 *
 * accuracy = (float)correct / dataset.count
 *
 * IMPLEMENTATION NOTES:
 * - network_predict returns predicted class (0-9)
 * - It calls forward propagation and returns argmax of output
 * - This function is used to evaluate on test set after training
 *
 * TYPICAL OUTPUT:
 *   printf("Accuracy: %.2f%% (%d/%d correct)\n",
 *          accuracy * 100, correct, total);
 */

#include "nn/metrics.h"
#include <stdio.h>

/* TODO: Implement accuracy calculation */

/* TODO: Implement evaluation function */
