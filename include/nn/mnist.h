#include <stddef.h>

// Dimensions
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
