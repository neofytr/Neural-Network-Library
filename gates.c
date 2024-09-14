#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float learning_rate = 1e-2;
float epsilon = 1e-2;
size_t iterations = 100000;
size_t train_count = 4;

float sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float rand_float()
{
    return ((float)rand() / (float)RAND_MAX);
}

float cost(float (*train)[3], size_t train_count, float w1, float w2, float b)
{
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoidf(w1 * x1 + w2 * x2 + b);
        float d = y - train[i][2];
        result += d * d;
    }
    return result / train_count;
}

void gradient_descent(float (*train)[3], size_t train_count, float *w1, float *w2, float *b, float rate, float eps, size_t iterations)
{
    float c = cost(train, train_count, *w1, *w2, *b);

    for (size_t i = 0; i < iterations; i++)
    {
        float dw1 = (cost(train, train_count, *w1 + eps, *w2, *b) - c) / eps;
        float dw2 = (cost(train, train_count, *w1, *w2 + eps, *b) - c) / eps;
        float db = (cost(train, train_count, *w1, *w2, *b + eps) - c) / eps;

        *w1 -= rate * dw1;
        *w2 -= rate * dw2;
        *b -= rate * db;

        c = cost(train, train_count, *w1, *w2, *b);
    }
}

void print_predictions(float (*train)[3], size_t train_count, float w1, float w2, float b)
{
    for (size_t i = 0; i < train_count; i++)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoidf(w1 * x1 + w2 * x2 + b);
        printf("%f OR %f = %f\n", x1, x2, y);
    }
}

/* int main()
{
    srand(time(NULL));

    size_t train_count = 4;
    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();

    return 0;
}
 */