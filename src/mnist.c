#include "nn/mnist.h"
#include <png.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>

size_t count_files(const char *path) {
    DIR *dir = opendir(path);
    if (!dir) {
        return 0;
    }

    size_t count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        count++;
    }

    closedir(dir);

    return count;
}

int mnist_load_png(const char *path, float *out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        return 0;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        return 0;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        return 0;
    }
    
    if (setjmp(png_jmpbuf(png))) {
        return 0;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

}

/* TODO: Implement mnist_load_png(filepath) - load single PNG to float array */
mnist_dataset_t *mnist_load_dataset(const char *path, char *mode) {
    mnist_dataset_t dataset;
    size_t total = 0;
    for (int label = 0; label < MNIST_NUM_CLASSES; label++) {
        char dir_path[256];
        snprintf(dir_path, sizeof(dir_path), "%s/%s/%d", path, mode, label);
        total += count_files(dir_path);
    }

    dataset.images = malloc(total * 784 * sizeof(float));
    dataset.labels = malloc(total * sizeof(uint8_t));
    dataset.count = total;

    int index = 0;
    for (int label = 0; label < MNIST_IMAGE_SIZE; )
}


/* TODO: Implement mnist_load_dataset(base_path, "train" or "test") */

/* TODO: Implement mnist_free_dataset(dataset) */

/* TODO: Implement mnist_shuffle(dataset) - Fisher-Yates shuffle */
