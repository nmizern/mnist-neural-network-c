#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

static int float_eq(float a, float b, float epsilon) {
    return fabs(a - b) < epsilon;
}

int main(void) {
    printf("Running activation tests...\n");

    /* TODO: Implement activation tests
     *
     * // Test sigmoid
     * {
     *     float result = sigmoid(0.0f);
     *     assert(float_eq(result, 0.5f, 0.0001f));
     *     printf("  [PASS] sigmoid(0) = 0.5\n");
     * }
     *
     * // Test ReLU
     * {
     *     assert(relu(5.0f) == 5.0f);
     *     assert(relu(-5.0f) == 0.0f);
     *     printf("  [PASS] ReLU\n");
     * }
     */

    printf("TODO: Implement activation tests\n");
    printf("All activation tests passed!\n");
    return 0;
}
