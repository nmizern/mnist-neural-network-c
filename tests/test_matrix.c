#include "nn/matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

/* Utilitaire : vérifier si deux flottants sont approximativement égaux */
static int float_eq(float a, float b, float epsilon) {
    return fabs(a - b) < epsilon;
}

int main(void) {
    printf("Running matrix tests...\n");

    /* Test création matrice */
    {
        Matrix *m = create_matrix(2, 3);
        assert(m != NULL);
        assert(m->rows == 2);
        assert(m->cols == 3);
        free_matrix(m);
        printf("  [PASS] Matrix creation\n");
    }

    /* Test dot product */
    {
        Vector *v1 = create_vector(3);
        Vector *v2 = create_vector(3);
        v1->data[0] = 1.0f;
        v1->data[1] = 2.0f;
        v1->data[2] = 3.0f;
        v2->data[0] = 4.0f;
        v2->data[1] = 5.0f;
        v2->data[2] = 6.0f;
        
        float result = dot_product(v1, v2);
        assert(float_eq(result, 32.0f, 1e-6));
        
        free_vector(v1);
        free_vector(v2);
        printf("  [PASS] Dot product\n");
    }
    
    /* Test addition de vecteurs */
    {
        Vector *v1 = create_vector(2);
        Vector *v2 = create_vector(2);
        v1->data[0] = 1.0f;
        v1->data[1] = 2.0f;
        v2->data[0] = 3.0f;
        v2->data[1] = 4.0f;
        
        Vector *result = add_vectors(v1, v2);
        assert(float_eq(result->data[0], 4.0f, 1e-6));
        assert(float_eq(result->data[1], 6.0f, 1e-6));
        
        free_vector(v1);
        free_vector(v2);
        free_vector(result);
        printf("  [PASS] Vector addition\n");
    }
    
    /* Test soustraction de vecteurs */
    {
        Vector *v1 = create_vector(2);
        Vector *v2 = create_vector(2);
        v1->data[0] = 5.0f;
        v1->data[1] = 7.0f;
        v2->data[0] = 3.0f;
        v2->data[1] = 2.0f;
        
        Vector *result = subtract_vectors(v1, v2);
        assert(float_eq(result->data[0], 2.0f, 1e-6));
        assert(float_eq(result->data[1], 5.0f, 1e-6));
        
        free_vector(v1);
        free_vector(v2);
        free_vector(result);
        printf("  [PASS] Vector subtraction\n");
    }
    
    /* Test multiplication element par element */
    {
        Vector *v1 = create_vector(2);
        Vector *v2 = create_vector(2);
        v1->data[0] = 2.0f;
        v1->data[1] = 3.0f;
        v2->data[0] = 4.0f;
        v2->data[1] = 5.0f;
        
        Vector *result = multiply_vectors(v1, v2);
        assert(float_eq(result->data[0], 8.0f, 1e-6));
        assert(float_eq(result->data[1], 15.0f, 1e-6));
        
        free_vector(v1);
        free_vector(v2);
        free_vector(result);
        printf("  [PASS] Vector element-wise multiplication\n");
    }
    
    /* Test transposition */
    {
        Matrix *m = create_matrix(2, 3);
        m->data[0] = 1.0f; m->data[1] = 2.0f; m->data[2] = 3.0f;
        m->data[3] = 4.0f; m->data[4] = 5.0f; m->data[5] = 6.0f;
        
        Matrix *t = transpose_matrix(m);
        assert(t->rows == 3);
        assert(t->cols == 2);
        assert(float_eq(t->data[0], 1.0f, 1e-6));
        assert(float_eq(t->data[1], 4.0f, 1e-6));
        assert(float_eq(t->data[2], 2.0f, 1e-6));
        
        free_matrix(m);
        free_matrix(t);
        printf("  [PASS] Matrix transpose\n");
    }
    
    /* Test copie de vecteur */
    {
        Vector *v1 = create_vector(3);
        v1->data[0] = 1.0f;
        v1->data[1] = 2.0f;
        v1->data[2] = 3.0f;
        
        Vector *v2 = create_vector(3);
        copy_vector(v1, v2);
        
        assert(float_eq(v2->data[0], 1.0f, 1e-6));
        assert(float_eq(v2->data[1], 2.0f, 1e-6));
        assert(float_eq(v2->data[2], 3.0f, 1e-6));
        
        free_vector(v1);
        free_vector(v2);
        printf("  [PASS] Vector copy\n");
    }

    printf("All matrix tests passed!\n");
    return 0;
}
