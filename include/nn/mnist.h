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
//size_t count_files(const char *base_path); moved static 

//done
//void swap_img(float *a, float *b); moved to static

//done
//void swap_labels(uint8_t *a, uint8_t *b);moved to static

//done 
int mnist_load_png(const char *path, float *out);

//done
mnist_dataset_t *mnist_load_dataset(const char *base_path, const char *mode);

//done
void mnist_free_dataset(mnist_dataset_t *dataset);

//done
void mnist_shuffle(mnist_dataset_t *dataset);

#endif 
