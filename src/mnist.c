#include "nn/mnist.h"
#include <png.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>

/* TODO: Implement mnist_load_png(filepath) - load single PNG to float array */
mnist_dataset_t *mnist_load_dataset(const char *path, char *mode) {
    for (int label = 0; label < MNIST_NUM_CLASSES; label++) {
        char dir_path[256];
        snprintf(dir_path, sizeof(dir_path), "%s/%s/%d", path, mode, label);

        DIR *dir = opendir(dir_path);
        if (!dir) {
            perror("Failed to open directory");
            return;
        }

        




    }
}


/* TODO: Implement mnist_load_dataset(base_path, "train" or "test") */

/* TODO: Implement mnist_free_dataset(dataset) */

/* TODO: Implement mnist_shuffle(dataset) - Fisher-Yates shuffle */
