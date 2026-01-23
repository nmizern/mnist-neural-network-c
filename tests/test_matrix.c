#include "nn/matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

/* Helper: check if two floats are approximately equal */
static int float_eq(float a, float b, float epsilon) {
    return fabs(a - b) < epsilon;
}

int main(void) {
    printf("Running matrix tests...\n");

    Matrix *m = create_matrix(2, 3);
    assert(m != NULL);
    assert(m->rows == 2);
    assert(m->cols == 3);
    printf("  [PASS] Matrix create/destroy\n");

    Vector *v1 = create_vector(2);
    Vector *v2 = create_vector(2);
    v1->data[0] = 1.0f;
    v1->data[1] = 2.0f;
    v2->data[0] = 3.0f;
    v2->data[1] = 4.0f;
    Vector *result = create_vector(2);
    mat_vec_mul(m, v1, result);
    assert(float_eq(result->data[0], 11.0f, 1e-6));
    assert(float_eq(result->data[1], 10.0f, 1e-6));
    free_matrix(m);
    free_vector(v1);
    free_vector(v2);

    printf("All matrix tests passed!\n");
    return 0;
}
