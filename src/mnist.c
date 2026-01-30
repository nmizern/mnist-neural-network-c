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
mnist_dataset_t MnistLoadDataset(const char *path, char *mode) {
    for (int label = 0; label < MNIST_NUM_CLASSES; label++) {
        char dir_path[256];
        snprintf(dir_path, sizeof(dir_path), "%s/%s/%d", path, mode, label);

        FILE *fp = fopen(dir_path, "rb");
        if (!fp) {

            

            
        }

    }
}


/* TODO: Implement mnist_load_dataset(base_path, "train" or "test") */

/* TODO: Implement mnist_free_dataset(dataset) */

/* TODO: Implement mnist_shuffle(dataset) - Fisher-Yates shuffle */
