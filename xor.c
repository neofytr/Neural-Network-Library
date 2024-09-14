/* #include <stdio.h>
#include "gates.c"
#include <string.h>

float or_train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

float and_train[][3] = {
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
};

float nand_train[][3] = {
    {0, 0, 1},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

float xor_train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

typedef struct
{
    float or_w1;
    float or_w2;
    float or_b;

    float nand_w1;
    float nand_w2;
    float nand_b;

    float and_w1;
    float and_w2;
    float and_b;
} xor ;

xor*x;

void increment_field(xor*x, const char *gate, const char *param)
{
    if (strcmp(gate, "or") == 0)
    {
        if (strcmp(param, "w1") == 0)
        {
            x->or_w1 += epsilon;
        }
        else if (strcmp(param, "w2") == 0)
        {
            x->or_w2 += epsilon;
        }
        else if (strcmp(param, "b") == 0)
        {
            x->or_b += epsilon;
        }
    }
    else if (strcmp(gate, "nand") == 0)
    {
        if (strcmp(param, "w1") == 0)
        {
            x->nand_w1 += epsilon;
        }
        else if (strcmp(param, "w2") == 0)
        {
            x->nand_w2 += epsilon;
        }
        else if (strcmp(param, "b") == 0)
        {
            x->nand_b += epsilon;
        }
    }
    else if (strcmp(gate, "and") == 0)
    {
        if (strcmp(param, "w1") == 0)
        {
            x->and_w1 += epsilon;
        }
        else if (strcmp(param, "w2") == 0)
        {
            x->and_w2 += epsilon;
        }
        else if (strcmp(param, "b") == 0)
        {
            x->and_b += epsilon;
        }
    }
}

void undo_increment_field(xor*x, const char *gate, const char *param)
{
    if (strcmp(gate, "or") == 0)
    {
        if (strcmp(param, "w1") == 0)
        {
            x->or_w1 -= epsilon;
        }
        else if (strcmp(param, "w2") == 0)
        {
            x->or_w2 -= epsilon;
        }
        else if (strcmp(param, "b") == 0)
        {
            x->or_b -= epsilon;
        }
    }
    else if (strcmp(gate, "nand") == 0)
    {
        if (strcmp(param, "w1") == 0)
        {
            x->nand_w1 -= epsilon;
        }
        else if (strcmp(param, "w2") == 0)
        {
            x->nand_w2 -= epsilon;
        }
        else if (strcmp(param, "b") == 0)
        {
            x->nand_b -= epsilon;
        }
    }
    else if (strcmp(gate, "and") == 0)
    {
        if (strcmp(param, "w1") == 0)
        {
            x->and_w1 -= epsilon;
        }
        else if (strcmp(param, "w2") == 0)
        {
            x->and_w2 -= epsilon;
        }
        else if (strcmp(param, "b") == 0)
        {
            x->and_b -= epsilon;
        }
    }
}

float forward(xor*m, float x1, float x2)
{
    float a = sigmoidf(m->or_w1 * (x1) + m->or_w2 * (x2) + m->or_b);
    float b = sigmoidf(m->nand_w1 * (x1) + m->and_w2 * (x2) + m->nand_b);

    return sigmoidf(m->and_w1 * a + m->and_w2 * b + m->and_b);
}

float cost_forward(float (*train)[3], xor*m)
{
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(m, x1, x2);
        float d = y - train[i][2];
        result += d * d;
    }
    return result / train_count;
}

float train(float (*data)[3], xor*x)
{
    for (size_t i = 0; i < 10000000; i++)
    {
        float c = cost_forward(xor_train, x);

        increment_field(x, "or", "w1");
        float dor_w1 = (cost_forward(xor_train, x) - c) / epsilon;
        undo_increment_field(x, "or", "w2");

        increment_field(x, "or", "w2");
        float dor_w2 = (cost_forward(xor_train, x) - c) / epsilon;
        undo_increment_field(x, "or", "w2");

        increment_field(x, "or", "b");
        float dor_b = (cost_forward(xor_train, x) - c) / epsilon;
        undo_increment_field(x, "or", "b");

        float dnand_w1, dnand_w2, dnand_b;
        increment_field(x, "nand", "w1");
        dnand_w1 = (cost_forward(xor_train, x) - c) / epsilon;
        undo_increment_field(x, "nand", "w1");

        increment_field(x, "nand", "w2");
        dnand_w2 = (cost_forward(xor_train, x) - c) / epsilon;
        undo_increment_field(x, "nand", "w2");

        increment_field(x, "nand", "b");
        dnand_b = (cost_forward(xor_train, x) - c) / epsilon;
        undo_increment_field(x, "nand", "b");

        float dand_w1, dand_w2, dand_b;
        increment_field(x, "and", "w1");
        dand_w1 = (cost_forward(xor_train, x) - c) / epsilon;
        undo_increment_field(x, "and", "w1");

        increment_field(x, "and", "w2");
        dand_w2 = (cost_forward(xor_train, x) - c) / epsilon;
        undo_increment_field(x, "and", "w2");

        increment_field(x, "and", "b");
        dand_b = (cost_forward(xor_train, x) - c) / epsilon;
        undo_increment_field(x, "and", "b");

        x->or_w1 -= learning_rate * dor_w1;
        x->or_w2 -= learning_rate * dor_w2;
        x->or_b -= learning_rate * dor_b;

        x->nand_w1 -= learning_rate * dnand_w1;
        x->nand_w2 -= learning_rate * dnand_w2;
        x->nand_b -= learning_rate * dnand_b;

        x->and_w1 -= learning_rate * dand_w1;
        x->and_w2 -= learning_rate * dand_w2;
        x->and_b -= learning_rate * dand_b;
    }
}

int main()
{
    x = malloc(sizeof(xor));

    x->or_w1 = rand_float();
    x->or_w2 = rand_float();
    x->or_b = rand_float();

    x->nand_w1 = rand_float();
    x->nand_w2 = rand_float();
    x->nand_b = rand_float();

    x->and_w1 = rand_float();
    x->and_w2 = rand_float();
    x->and_b = rand_float();
    train(xor_train, x);

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            float a = sigmoidf(x->or_w1 * i + x->or_w2 * j + x->or_b);
            float b = sigmoidf(x->nand_w1 * i + x->and_w2 * j + x->nand_b);

            printf("%zu XOR %zu = %f\n", i, j, sigmoidf(x->and_w1 * a + x->and_w2 * b + x->and_b));
        }
    }
} */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define PAIR_COUNT 4
#define MAX_ITER 1000000
#define LEARNING_RATE 1e-2
#define EPSILON 1e-1

typedef struct {
    float or_w1;
    float or_w2;
    float or_b;

    float nand_w1;
    float nand_w2;
    float nand_b;

    float and_w1;
    float and_w2;
    float and_b;
} XOR;

float sigmoidf(float x) {
    return (1.0f / (1.0f + expf(-x)));
}

float forward(XOR*m, float x1, float x2)
{
    float a = sigmoidf(m->or_w1 * (x1) + m->or_w2 * (x2) + m->or_b);
    float b = sigmoidf(m->nand_w1 * (x1) + m->and_w2 * (x2) + m->nand_b);

    return sigmoidf(m->and_w1 * a + m->and_w2 * b + m->and_b);
}

float xor_train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

float cost_forward(float (*train)[3], XOR *x) {
    float result = 0.0f;
    for (size_t i = 0; i < PAIR_COUNT; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(x, x1, x2);
        float d = y - train[i][2];
        result += d * d;
    }
    return result / PAIR_COUNT;
}

void gradient_descent(XOR *x) {
    float eps = EPSILON;
    float rate = LEARNING_RATE;

    float original_or_w1, original_or_w2, original_or_b;
    float original_nand_w1, original_nand_w2, original_nand_b;
    float original_and_w1, original_and_w2, original_and_b;

    for (size_t i = 0; i < MAX_ITER; i++) {
        original_or_w1 = x->or_w1;
        original_or_w2 = x->or_w2;
        original_or_b = x->or_b;

        original_nand_w1 = x->nand_w1;
        original_nand_w2 = x->nand_w2;
        original_nand_b = x->nand_b;

        original_and_w1 = x->and_w1;
        original_and_w2 = x->and_w2;
        original_and_b = x->and_b;

        float c = cost_forward(xor_train, x);

        x->or_w1 += eps;
        float c1 = cost_forward(xor_train, x);
        x->or_w1 = original_or_w1;

        x->or_w2 += eps;
        float c2 = cost_forward(xor_train, x);
        x->or_w2 = original_or_w2;

        x->or_b += eps;
        float c3 = cost_forward(xor_train, x);
        x->or_b = original_or_b;

        float dor_w1 = (c1 - c) / eps;
        float dor_w2 = (c2 - c) / eps;
        float dor_b = (c3 - c) / eps;

        x->or_w1 = original_or_w1;
        x->or_w2 = original_or_w2;
        x->or_b = original_or_b;

        x->nand_w1 += eps;
        c1 = cost_forward(xor_train, x);
        x->nand_w1 = original_nand_w1;

        x->nand_w2 += eps;
        c2 = cost_forward(xor_train, x);
        x->nand_w2 = original_nand_w2;

        x->nand_b += eps;
        c3 = cost_forward(xor_train, x);
        x->nand_b = original_nand_b;

        float dnand_w1 = (c1 - c) / eps;
        float dnand_w2 = (c2 - c) / eps;
        float dnand_b = (c3 - c) / eps;

        x->nand_w1 = original_nand_w1;
        x->nand_w2 = original_nand_w2;
        x->nand_b = original_nand_b;

        x->and_w1 += eps;
        c1 = cost_forward(xor_train, x);
        x->and_w1 = original_and_w1;

        x->and_w2 += eps;
        c2 = cost_forward(xor_train, x);
        x->and_w2 = original_and_w2;

        x->and_b += eps;
        c3 = cost_forward(xor_train, x);
        x->and_b = original_and_b;

        float dand_w1 = (c1 - c) / eps;
        float dand_w2 = (c2 - c) / eps;
        float dand_b = (c3 - c) / eps;

        x->and_w1 = original_and_w1;
        x->and_w2 = original_and_w2;
        x->and_b = original_and_b;

        x->or_w1 -= rate * dor_w1;
        x->or_w2 -= rate * dor_w2;
        x->or_b -= rate * dor_b;

        x->nand_w1 -= rate * dnand_w1;
        x->nand_w2 -= rate * dnand_w2;
        x->nand_b -= rate * dnand_b;

        x->and_w1 -= rate * dand_w1;
        x->and_w2 -= rate * dand_w2;
        x->and_b -= rate * dand_b;

        if (i % 10000 == 0) {
            printf("cost: %f\n", c);
        }
    }
}

int main() {
    XOR x = {0};
    gradient_descent(&x);
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            printf("%zu XOR %zu = %f\n", i, j, forward(&x, i, j));
        }
    }
    return 0;
}
