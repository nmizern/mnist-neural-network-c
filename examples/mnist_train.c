/**
 * @file mnist_train.c
 * @brief Full MNIST training pipeline
 *
 * This is the main example that demonstrates the complete workflow:
 * 1. Load MNIST dataset
 * 2. Create neural network
 * 3. Train on training set
 * 4. Evaluate on test set
 *
 * USAGE:
 *   ./mnist_train <path_to_mnist_pngs>
 *
 * The mnist-pngs directory should have structure:
 *   mnist-pngs/
 *   ├── train/
 *   │   ├── 0/, 1/, ..., 9/
 *   └── test/
 *       ├── 0/, 1/, ..., 9/
 *
 * RECOMMENDED HYPERPARAMETERS:
 * - Network: [784, 128, 64, 10] or [784, 256, 10]
 * - Learning rate: 0.01 - 0.1
 * - Epochs: 5-20
 * - Hidden activation: ReLU or Sigmoid
 * - Output activation: Sigmoid (with MSE loss)
 *
 * EXPECTED RESULTS:
 * - After 5 epochs: ~90% accuracy
 * - After 10 epochs: ~95% accuracy
 * - After 20 epochs: ~97% accuracy
 */

#include "neuralnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    printf("MNIST Neural Network Training\n");
    printf("==============================\n\n");

    if (argc < 2) {
        printf("Usage: %s <path_to_mnist_pngs>\n", argv[0]);
        printf("Example: %s ./data/mnist-pngs\n", argv[0]);
        return 1;
    }

    const char *data_path = argv[1];
    printf("Data path: %s\n\n", data_path);

    /* Seed random number generator for reproducibility */
    srand((unsigned int)time(NULL));

    /* TODO: Implement MNIST training
     *
     * Steps:
     *
     * 1. Load training dataset
     *    Dataset train_data = mnist_load(data_path, "train");
     *    printf("Loaded %d training samples\n", train_data.count);
     *
     * 2. Load test dataset
     *    Dataset test_data = mnist_load(data_path, "test");
     *    printf("Loaded %d test samples\n", test_data.count);
     *
     * 3. Create network
     *    int sizes[] = {784, 128, 10};
     *    Network *net = network_create(sizes, 3);
     *
     * 4. Training loop
     *    float learning_rate = 0.01;
     *    int epochs = 10;
     *
     *    for (int epoch = 0; epoch < epochs; epoch++) {
     *        train_epoch(net, train_data, learning_rate);
     *        float accuracy = evaluate(net, test_data);
     *        printf("Epoch %d: accuracy = %.2f%%\n", epoch+1, accuracy*100);
     *    }
     *
     * 5. Final evaluation
     *    float final_accuracy = evaluate(net, test_data);
     *    printf("Final accuracy: %.2f%%\n", final_accuracy*100);
     *
     * 6. Cleanup
     *    network_destroy(net);
     *    dataset_destroy(train_data);
     *    dataset_destroy(test_data);
     */

    printf("TODO: Implement MNIST training\n");

    return 0;
}
