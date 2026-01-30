/**
 * @file mnist.h
 * @brief MNIST dataset loading from PNG files using libpng
 *
 * Loads images from: https://github.com/rasbt/mnist-pngs
 *
 * Directory structure:
 *   mnist-pngs/train/0..9/*.png   (training images)
 *   mnist-pngs/test/0..9/*.png    (test images)
 *
 * Each image: 28x28 grayscale PNG -> 784 floats normalized to [0,1]
 */

#ifndef NN_MNIST_H
#define NN_MNIST_H

#include <stddef.h>
#include <stdint.h>

#define MNIST_IMAGE_SIZE   (28 * 28)  /* 784 */
#define MNIST_NUM_CLASSES  10

/* TODO: Dataset structure (images array, labels array, count) */
/* TODO: mnist_load_dataset(base_path, "train" or "test") */
/* TODO: mnist_free_dataset(dataset) */
/* TODO: mnist_shuffle(dataset) */

#endif /* NN_MNIST_H */
