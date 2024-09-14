#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PAIR_COUNT 4
float train[PAIR_COUNT][2] = {{1, 2}, {2, 4}, {3, 6}, {4, 8}};
float rand_float(void) { return (float)rand() / (float)RAND_MAX * 1.0f; }
void generate_random_pairs()
{
    for (size_t i = 0; i < PAIR_COUNT; i++)
    {
        train[i][0] = rand_float();
        train[i][1] = rand_float();
    }
}

float cost(float a, float b, float c)
{
    float result = 0.0f;
    for (size_t i = 0; i < PAIR_COUNT; i++)
    {
        float x = train[i][0];
        float y = x * x * a + b * x + c;
        float d = y - train[i][1];
        result += d * d;
    }
    result /= PAIR_COUNT;
    return result;
}

int main(int argc, char **argv)
{
    // y = a * x; a is the model parameter
    srand(time(NULL));
    // srand(69);
    // generate_random_pairs();
    float a = rand_float();
    float b = rand_float();
    float c = rand_float();

    float eps = 1e-3;
    float rate = 1e-3;

    printf("%f\n", cost(a, b, c));

    for (size_t i = 0; i < 1000000; i++)
    {
        float da = (cost(a + eps, b, c) - cost(a, b, c)) / eps;
        float db = (cost(a, b + eps, c) - cost(a, b, c)) / eps;
        float dc = (cost(a, b, c + eps) - cost(a, b, c)) / eps;
        a -= rate * da;
        b -= rate * db;
        c -= rate * dc;
        printf("cost: %f, a: %f, b = %f, c = %f\n", cost(a, b, c), a, b, c);
    }

    // printf("-------------\n");
    // printf("%f\n", a);

    return EXIT_SUCCESS;
}
