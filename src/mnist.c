/**
 * @file mnist.c
 * @brief MNIST dataset loading from PNG files using libpng
 *
 * Loads images from: https://github.com/rasbt/mnist-pngs
 * Directory structure: mnist-pngs/{train,test}/0..9/*.png
 * Each image: 28x28 grayscale PNG -> 784 floats normalized to [0,1]
 */

#include "nn/mnist.h"
#include <png.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* TODO: Implement mnist_load_png(filepath) - load single PNG to float array */

/* TODO: Implement mnist_load_dataset(base_path, "train" or "test") */

/* TODO: Implement mnist_free_dataset(dataset) */

/* TODO: Implement mnist_shuffle(dataset) - Fisher-Yates shuffle */
