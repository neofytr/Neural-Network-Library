#define NN_IMPLEMENTATION_
#define MODEL DIFF
#include "nn.h"

#define BITS 2

int main(void)
{
    //omp_set_num_threads(16);
    size_t n = 1 << BITS;
    size_t rows = n * n;
    Mat *ti = mat_alloc(rows, 2 * BITS);
    Mat *to = mat_alloc(rows, BITS + 1);
    for (size_t i = 0; i < ti->rows; i++)
    {
        size_t x = i / n;
        size_t y = i % n;
        size_t z = x + y;
        for (size_t j = 0; j < BITS; j++)
        {
            MAT_AT(ti, i, j) = (x>>j)&1;
            MAT_AT(ti,i, j + BITS) = (y>>j)&1;
            MAT_AT(to, i, j) = (z>>j)&1;
        }
        MAT_AT(to, i, BITS) = (z >= n);
    }

    //MAT_PRINT(ti);
    //MAT_PRINT(to);

    size_t arch[] = {2*BITS, 6, BITS + 1};
    NN *nn = nn_alloc(arch, ARRAY_LEN(arch));
    randomize_parameters_NN(nn, 0, 0);
    NN_PRINT(nn);

    learn(nn, 1e-1, 1e-1, 100000, ti, to);
}