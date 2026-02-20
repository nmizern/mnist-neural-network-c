#include "nn/mnist.h"
#include <png.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <time.h>

size_t count_files(const char *path) {
    DIR *dir = opendir(path);
    if (!dir) {
        return 0;
    }

    size_t count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        count++;
    }

    closedir(dir);

    return count;
}

static void swap_img(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

static void swap_labels(uint8_t *a, uint8_t *b) {
    uint8_t temp = *a;
    *a = *b;
    *b = temp;
}

int mnist_load_png(const char *path, float *out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        return 0;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        return 0;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        return 0;
    }
    
    if (setjmp(png_jmpbuf(png))) {
        fclose(fp);
        return 0;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);

    png_bytep *row_pointers = malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
    }
    png_read_image(png, row_pointers);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            out[y * width + x] = row_pointers[y][x] / 255.0f;
        }
    }

    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
    png_destroy_read_struct(&png, &info, NULL);

    fclose(fp);
    return 1;

}

mnist_dataset_t *mnist_load_dataset(const char *path, const char *mode) {
    mnist_dataset_t *dataset = malloc(sizeof(mnist_dataset_t));
    if (!dataset) return NULL;

    size_t total = 0;
    for (int label = 0; label < MNIST_NUM_CLASSES; label++) {
        char dir_path[256];
        snprintf(dir_path, sizeof(dir_path), "%s/%s/%d", path, mode, label);
        total += count_files(dir_path);
    }

    dataset->images = malloc(total * 784 * sizeof(float));
    if (!dataset->images) {
        free(dataset);
        return NULL;
    }

    dataset->labels = malloc(total * sizeof(uint8_t));
    if (!dataset->labels) {
        free(dataset->images);
        free(dataset);
        return NULL;
    }

    dataset->count = total;

    size_t index = 0;
    for (int label = 0; label < MNIST_NUM_CLASSES; label++) {
        char dir_path[256];
        snprintf(dir_path, sizeof(dir_path), "%s/%s/%d", path, mode, label);
        
        DIR *dir = opendir(dir_path);
        if (!dir) {
            continue;
        }

        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_name[0] == '.') continue;

            char filepath[512];
            snprintf(filepath, sizeof(filepath), "%s/%s", dir_path, entry->d_name);

            mnist_load_png(filepath, &dataset->images[index * MNIST_IMAGE_SIZE]);
            dataset->labels[index] = (uint8_t)label;
            index++;
        }
        closedir(dir);
    }

    return dataset;
}

void mnist_free_dataset(mnist_dataset_t *dataset) {
    if (!dataset) {
        return;
    }
    free(dataset->images);
    free(dataset->labels);
    free(dataset);
}

void mnist_shuffle(mnist_dataset_t *dataset) {
    for (size_t i = dataset->count - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        float *img_i = &dataset->images[i * MNIST_IMAGE_SIZE];
        float *img_j = &dataset->images[j * MNIST_IMAGE_SIZE];
        for (int k = 0; k < MNIST_IMAGE_SIZE; k++) {
            swap_img(&img_i[k], &img_j[k]);
        }

        swap_labels(&dataset->labels[i], &dataset->labels[j]);
        
        
    }
}
