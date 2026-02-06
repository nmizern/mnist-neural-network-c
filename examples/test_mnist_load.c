#include "neuralnet.h"
#include <stdio.h>
#include <stdlib.h>

//ALL IS GENERATED NOW IT'S JUST FOR TEST LOADING MNIST DATASET, NOT TRAINING

int main(void) {
    const char *base = "E:/proj/NN/examples/mnist-pngs";

    /* Step 1: Test count_files on a single directory */
    char dir0[256];
    snprintf(dir0, sizeof(dir0), "%s/train/0", base);
    printf("Step 1: count_files(\"%s\")\n", dir0);
    fflush(stdout);

    size_t n = count_files(dir0);
    printf("  => %zu files\n", n);
    fflush(stdout);

    /* Step 2: Test loading a single PNG */
    printf("Step 2: mnist_load_png on first file\n");
    fflush(stdout);

    float pixels[784];
    int ok = mnist_load_png(dir0, pixels);  /* intentionally wrong: dir, not file */
    printf("  => result (should be 0 for dir): %d\n", ok);
    fflush(stdout);

    /* Try an actual PNG file */
    char png_path[512];
    snprintf(png_path, sizeof(png_path), "%s/1.png", dir0);
    printf("Step 3: mnist_load_png(\"%s\")\n", png_path);
    fflush(stdout);

    ok = mnist_load_png(png_path, pixels);
    printf("  => result: %d\n", ok);
    if (ok) {
        printf("  => first 5 pixels: %.3f %.3f %.3f %.3f %.3f\n",
               pixels[0], pixels[1], pixels[2], pixels[3], pixels[4]);
    }
    fflush(stdout);

    /* Step 4: Load small dataset (just count) */
    printf("Step 4: mnist_load_dataset (train)\n");
    fflush(stdout);

    mnist_dataset_t *train = mnist_load_dataset(base, "train");
    if (!train) {
        printf("  => FAILED: returned NULL\n");
        fflush(stdout);
        return 1;
    }
    printf("  => OK: %zu samples, label[0]=%d\n", train->count, train->labels[0]);
    fflush(stdout);

    mnist_free_dataset(train);
    printf("Step 5: free OK\n");
    fflush(stdout);

    printf("\nAll tests passed!\n");
    return 0;
}
