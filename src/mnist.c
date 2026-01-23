/**
 * @file mnist.c
 * @brief MNIST dataset loading implementation
 *
 * Implement the functions declared in nn/mnist.h here.
 *
 * RECOMMENDED APPROACH - Using stb_image.h:
 *
 * 1. Download stb_image.h from:
 *    https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
 *
 * 2. Place it in include/third_party/stb_image.h
 *
 * 3. In ONE .c file (this one), do:
 *    #define STB_IMAGE_IMPLEMENTATION
 *    #include "third_party/stb_image.h"
 *
 * 4. Usage:
 *    int width, height, channels;
 *    unsigned char *img = stbi_load(path, &width, &height, &channels, 1);
 *    // channels=1 forces grayscale
 *    // img[i] is pixel value 0-255
 *    // Normalize: float_pixel = img[i] / 255.0f
 *    stbi_image_free(img);
 *
 * DIRECTORY ITERATION:
 * - On Windows: use _findfirst/_findnext or dirent.h (if available)
 * - On Linux/macOS: use dirent.h (opendir, readdir)
 * - Cross-platform: consider generating file list externally
 *
 * SIMPLE ALTERNATIVE:
 * Generate file paths programmatically:
 *   for label 0-9:
 *       for i = 0 to count:
 *           sprintf(path, "%s/%d/%05d.png", base_path, label, i)
 *           load image
 *
 * Note: You need to know how many images per class or detect end-of-files.
 */

#include "nn/mnist.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Uncomment after adding stb_image.h to include/third_party/ */
/* #define STB_IMAGE_IMPLEMENTATION */
/* #include "third_party/stb_image.h" */

/* TODO: Implement MNIST loading functions */
