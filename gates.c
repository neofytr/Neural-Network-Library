#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float rand_float()
{
    return ((float)rand() / (float)RAND_MAX);
}

void gradient_calculator(float (*train)[3], size_t train_count, float *dw1, float *dw2, float *db, float w1, float w2, float b)
{
    // Reset gradients
    float ddw1 = 0.0f;
    float ddw2 = 0.0f;
    float ddb = 0.0f;

    // Compute gradients
    for (size_t r = 0; r < train_count; r++)
    {
        float x1 = train[r][0];
        float x2 = train[r][1];
        float y_r = train[r][2];
        float y_dash_r = sigmoidf(x1 * w1 + x2 * w2 + b);

        float dr = y_dash_r * (1 - y_dash_r); // Derivative of sigmoid
        float yr = (y_dash_r - y_r);          // Error

        ddb += 2 * (yr * dr);
        ddw1 += 2 * (x1 * yr * dr);
        ddw2 += 2 * (x2 * yr * dr);
    }

    // Average the gradients
    ddb /= train_count;
    ddw1 /= train_count;
    ddw2 /= train_count;

    *db = ddb;
    *dw1 = ddw1;
    *dw2 = ddw2;
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
    for (size_t i = 0; i < iterations; i++)
    {
        float dw1 = 0.0f;
        float dw2 = 0.0f;
        float db = 0.0f;

        gradient_calculator(train, train_count, &dw1, &dw2, &db, *w1, *w2, *b);
        *w1 -= rate * dw1;
        *w2 -= rate * dw2;
        *b -= rate * db;

        // Log the cost every few iterations
        if (i % 100 == 0)
        {
            float c = cost(train, train_count, *w1, *w2, *b);
            printf("Iteration %zu: Cost = %f\n", i, c);
        }
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

int main()
{
    srand(time(NULL));

    float or_train[][3] = {
        {0, 0, 0},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 1},
    };

    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();

    gradient_descent(or_train, 4, &w1, &w2, &b, 1e-1, 1e-1, 100000);
    print_predictions(or_train, 4, w1, w2, b);

    return 0;
}
