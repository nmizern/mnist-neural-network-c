/**
 * @file mnist.h
 * @brief MNIST dataset loading from PNG files
 *
 * This module handles loading the MNIST dataset from PNG image files.
 * The dataset source: https://github.com/rasbt/mnist-pngs
 *
 * DATASET STRUCTURE (mnist-pngs repository):
 * mnist-pngs/
 * ├── train/
 * │   ├── 0/          # All images of digit 0
 * │   │   ├── 00000.png
 * │   │   ├── 00001.png
 * │   │   └── ...
 * │   ├── 1/          # All images of digit 1
 * │   └── ...
 * └── test/
 *     ├── 0/
 *     ├── 1/
 *     └── ...
 *
 * REQUIRED FUNCTIONALITY:
 *
 * 1. Load single PNG image as normalized float array
 *    - Input: file path
 *    - Output: array of 784 floats (28x28), values in [0, 1]
 *    - Normalization: pixel_value / 255.0
 *
 * 2. Load entire dataset (train or test)
 *    - Iterate through folders 0-9
 *    - Load all images with their labels
 *    - Return arrays of images and labels
 *
 * 3. Dataset structure for training
 *    - Images array: N x 784 floats
 *    - Labels array: N integers (0-9)
 *    - Total count
 *
 * PNG LOADING OPTIONS:
 * - Use stb_image.h (single header, no dependencies) - RECOMMENDED
 * - Manual PNG parsing (complex, not recommended)
 * - Convert to simpler format first (e.g., raw binary)
 *
 * stb_image.h can be downloaded from:
 * https://github.com/nothings/stb/blob/master/stb_image.h
 *
 * MEMORY CONSIDERATIONS:
 * - MNIST training set: 60,000 images x 784 floats = ~180 MB
 * - Consider loading in batches for memory-constrained systems
 */

#ifndef NN_MNIST_H
#define NN_MNIST_H

#include <stddef.h>

/* Image dimensions */
#define MNIST_IMAGE_WIDTH  28
#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_SIZE   (MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT)
#define MNIST_NUM_CLASSES  10

/* TODO: Define dataset structure */
/* Should contain: images array, labels array, count */

/* TODO: Load single image from PNG file */
/* Returns normalized float array [0, 1] */

/* TODO: Load entire train or test dataset from directory */
/* Input: path to "train" or "test" folder */
/* Output: populated dataset structure */

/* TODO: Free dataset memory */

/* TODO: Shuffle dataset (for training) */

/* TODO: Get batch from dataset (optional, for batch training) */

#endif /* NN_MNIST_H */
