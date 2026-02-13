#include "neuralnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "nn/mnist.h"
#include "nn/network.h"

#define MNIST_IMAGE_SIZE 784
#define MNIST_NUM_CLASSES 10

Network mnist_nn() {
    size_t layer_sizes[] = {MNIST_IMAGE_SIZE, 128, 64, MNIST_NUM_CLASSES};
    size_t num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
    return *nn_network_create(layer_sizes, num_layers);
}

// Returns the average loss for the epoch
float train_epoch(Network *network, mnist_dataset_t *train_data, float learning_rate) {
    float total_loss = 0.0f;
    
    mnist_shuffle(train_data);
    
    // TODO: Loop through all training samples
    for (size_t i = 0; i < train_data->count; i++) {
        float *image = &train_data->images[i * MNIST_IMAGE_SIZE];
        uint8_t label = train_data->labels[i];
        Vector *input = create_vector(MNIST_IMAGE_SIZE);
        mnist_load_png(image, input)
        // Hint: Use nn_network_forward() and nn_network_backward()
    }
    
    return total_loss / train_data->count;
}

// TODO: Implement this function to evaluate the network on test data
// Returns the accuracy (percentage of correct predictions)
float evaluate_network(Network *network, mnist_dataset_t *test_data) {
    size_t correct = 0;
    
    // TODO: Loop through all test samples
    for (size_t i = 0; i < test_data->count; i++) {
        // TODO: 1. Create input vector from image data
        // TODO: 2. Use nn_network_predict() to get predicted class
        // TODO: 3. Compare with actual label and count correct predictions
    }
    
    return (float)correct / test_data->count * 100.0f;
}

// Helper function: Convert label to one-hot vector
Vector* label_to_onehot(uint8_t label) {
    Vector *target = create_vector(MNIST_NUM_CLASSES);
    for (size_t i = 0; i < MNIST_NUM_CLASSES; i++) {
        target->data[i] = (i == label) ? 1.0f : 0.0f;
    }
    return target;
}

// Helper function: Convert image array to vector
Vector* image_to_vector(const float *image_data) {
    Vector *vec = create_vector(MNIST_IMAGE_SIZE);
    memcpy(vec->data, image_data, MNIST_IMAGE_SIZE * sizeof(float));
    return vec;
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    printf("MNIST Neural Network Training\\n");
    printf("==============================\\n\\n");

    if (argc < 2) {
        printf("Usage: %s <path_to_mnist_pngs>\\n", argv[0]);
        printf("\\nThe directory should contain:\\n");
        printf("  train/0/, train/1/, ..., train/9/\\n");
        printf("  test/0/, test/1/, ..., test/9/\\n");
        printf("\\nDataset: https://github.com/rasbt/mnist-pngs\\n");
        return 1;
    }

    const char *data_path = argv[1];
    printf("Data path: %s\\n\\n", data_path);

    // ========================================
    // 1. Load MNIST Dataset
    // ========================================
    printf("Loading training data...\\n");
    mnist_dataset_t *train_data = mnist_load_dataset(data_path, "train");
    if (!train_data) {
        fprintf(stderr, "Failed to load training data\\n");
        return 1;
    }
    printf("Loaded %zu training samples\\n", train_data->count);

    printf("Loading test data...\\n");
    mnist_dataset_t *test_data = mnist_load_dataset(data_path, "test");
    if (!test_data) {
        fprintf(stderr, "Failed to load test data\\n");
        mnist_free_dataset(train_data);
        return 1;
    }
    printf("Loaded %zu test samples\\n\\n", test_data->count);

    // ========================================
    // 2. Create Neural Network
    // ========================================
    // Network architecture: 784 -> 128 -> 64 -> 10
    // TODO: Experiment with different architectures!
    size_t layer_sizes[] = {MNIST_IMAGE_SIZE, 128, 64, MNIST_NUM_CLASSES};
    size_t num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
    
    Network *network = nn_network_create(layer_sizes, num_layers);
    nn_network_print(network);
    printf("\\n");

    // ========================================
    // 3. Training Configuration
    // ========================================
    const int NUM_EPOCHS = 10;  // TODO: Adjust as needed
    const float LEARNING_RATE = 0.01f;  // TODO: Experiment with this value
    
    printf("Training configuration:\\n");
    printf("  Epochs: %d\\n", NUM_EPOCHS);
    printf("  Learning rate: %.4f\\n\\n", LEARNING_RATE);

    // ========================================
    // 4. Training Loop
    // ========================================
    printf("Starting training...\\n");
    printf("--------------------\\n");
    
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        // TODO: Uncomment and implement train_epoch()
        // float avg_loss = train_epoch(network, train_data, LEARNING_RATE);
        
        // TODO: Uncomment to evaluate after each epoch
        // float accuracy = evaluate_network(network, test_data);
        
        // TODO: Print progress
        // printf("Epoch %d/%d - Loss: %.4f - Accuracy: %.2f%%\\n", 
        //        epoch + 1, NUM_EPOCHS, avg_loss, accuracy);
        
        printf("Epoch %d/%d - [TODO: Implement training]\\n", epoch + 1, NUM_EPOCHS);
    }
    
    printf("\\nTraining complete!\\n\\n");

    // ========================================
    // 5. Final Evaluation
    // ========================================
    // TODO: Uncomment when evaluate_network() is implemented
    // printf("Final test accuracy: %.2f%%\\n", evaluate_network(network, test_data));

    // ========================================
    // 6. Save the trained model (optional)
    // ========================================
    // TODO: Uncomment to save the model
    // nn_network_save(network, "mnist_model.bin");

    // ========================================
    // 7. Test on individual samples
    // ========================================
    printf("\\nTesting on first 5 test samples:\\n");
    for (int i = 0; i < 5 && i < (int)test_data->count; i++) {
        Vector *input = image_to_vector(&test_data->images[i * MNIST_IMAGE_SIZE]);
        size_t predicted_class;
        nn_network_predict(network, input, &predicted_class);
        
        printf("Sample %d: True label = %d, Predicted = %zu\\n", 
               i, test_data->labels[i], predicted_class);
        
        free_vector(input);
    }

    // ========================================
    // 8. Cleanup
    // ========================================
    nn_network_free(network);
    mnist_free_dataset(train_data);
    mnist_free_dataset(test_data);

    printf("\\nDone!\\n");
    return 0;
}
