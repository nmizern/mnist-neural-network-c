#ifndef NN_MNIST_H
#define NN_MNIST_H

#include <stddef.h>
#include <stdint.h>

#define MNIST_IMAGE_SIZE   (28 * 28)
#define MNIST_NUM_CLASSES  10

typedef struct {
    float   *images;
    uint8_t *labels;                     
    size_t   count;                 
} mnist_dataset_t;

//done
mnist_dataset_t *mnist_load_dataset(const char *base_path, const char *mode);

void mnist_free_dataset(mnist_dataset_t *dataset);

void mnist_shuffle(mnist_dataset_t *dataset);

//done
size_t count_files(const char *base_path);

#endif 
