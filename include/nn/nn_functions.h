#ifndef NN_FUNCTIONS_H
#define NN_FUNCTIONS_H

#include <stddef.h>


float nn_sigmoid(float x);

float nn_sigmoid_deriv(float output);

float nn_relu(float x);

float nn_relu_deriv(float x);

void nn_activate(float *vec, size_t n, float (*func)(float));

void nn_one_hot(size_t label, size_t num_classes, float *result);

float nn_mse_loss(const float *pred, const float *target, size_t n);

void nn_mse_gradient(const float *pred, const float *target, size_t n, float *grad);

#endif
