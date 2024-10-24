/* #define NN_IMPLEMENTATION_
#define ELEMENT_TYPE double
#define ARRAY_LEN(xs) (sizeof(xs) / sizeof(xs[0]))
#define SIZE_OF_TRAINING_DATA 4
#include <string.h>

#include "nn.h"

typedef struct
{
    size_t arch_count;
    Mat **ws;
    Mat **bs;
    Mat **as; // The amount of activations is count + 1
} NN;

NN *nn_alloc(size_t *arch, size_t arch_count)
{
    NN *nn = malloc(sizeof(NN));
    if (!nn)
    {
        perror("nn_alloc");
        exit(EXIT_FAILURE);
    }

    nn->ws = malloc(sizeof(Mat *) * (arch_count - 1));
    if (!(nn->ws))
    {
        perror("nn_alloc");
        exit(EXIT_FAILURE);
    }

    nn->bs = malloc(sizeof(Mat *) * (arch_count - 1));
    if (!(nn->bs))
    {
        perror("nn_alloc");
        exit(EXIT_FAILURE);
    }

    nn->as = malloc(sizeof(Mat *) * (arch_count));
    if (!(nn->as))
    {
        perror("nn_alloc");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < arch_count; i++)
    {
        if (i == 0)
        {
            nn->ws[i] = NULL;
            nn->bs[i] = NULL;
            nn->as[i] = mat_alloc(1, arch[i]);
        }
        else
        {
            nn->as[i] = mat_alloc(1, arch[i]);
            nn->ws[i] = mat_alloc(arch[i - 1], arch[i]);
            nn->bs[i] = mat_alloc(1, arch[i]);
        }
    }

    nn->arch_count = arch_count;
    return nn;
}

void randomize_parameters_NN(NN *nn)
{
    size_t arch_count = nn->arch_count;

    for (size_t i = 1; i < arch_count; i++)
    {
        mat_rand(nn->ws[i], 1, 0);
        mat_rand(nn->bs[i], 1, 0);
    }
}

void forward_NN(NN *nn)
{
    size_t arch_count = nn->arch_count;
    for (size_t i = 1; i < arch_count; i++)
    {
        mat_dot(nn->as[i], nn->as[i - 1], nn->ws[i]);
        mat_sum(nn->as[i], nn->bs[i]);
        mat_sig(nn->as[i]);
    }
}

ELEMENT_TYPE cost_NN(NN *nn, Mat *training_input, Mat *training_output)
{
    size_t arch_count = nn->arch_count;
    assert(training_input->rows == training_output->rows);
    assert(nn->as[nn->arch_count - 1]->cols == training_output->cols);

    size_t rows = training_input->rows;
    size_t cols = training_output->cols;

    ELEMENT_TYPE result = (ELEMENT_TYPE)0;
    for (size_t i = 0; i < rows; i++)
    {
        Mat x = mat_row(training_input, i);
        Mat y = mat_row(training_output, i);
        Mat *y_ = &y;

        *(nn->as[0]) = x;
        forward_NN(nn);

        // MAT_PRINT(&NN_input);
        // MAT_PRINT(&NN_output);
        for (size_t j = 0; j < cols; j++)
        {
            ELEMENT_TYPE d = MAT_AT(nn->as[arch_count - 1], i, j) - MAT_AT(y_, i, j);
            // printf("%f\n", d);
            result += d * d;
        }
    }
    // printf("%f\n", result/rows);
    return (result / rows);
}

void diff(NN *nn, Mat *training_input, Mat *training_output, ELEMENT_TYPE eps, ELEMENT_TYPE learning_rate, Mat *temp_para, ELEMENT_TYPE c)
{
    ELEMENT_TYPE saved;
    size_t rows = temp_para->rows, cols = temp_para->cols;
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            saved = MAT_AT(temp_para, i, j);
            *MAT_AT_POINTER(temp_para, i, j) += eps;
            // printf("%f\n", r);
            ELEMENT_TYPE d = (cost_NN(nn, training_input, training_output) - c) / eps;
            *MAT_AT_POINTER(temp_para, i, j) = saved;
            // printf("%f\n", MAT_AT(temp_para, i, j));
            *MAT_AT_POINTER(temp_para, i, j) -= d * learning_rate;
            // printf("%f\n", MAT_AT(temp_para, i, j));
        }
    }
}

void learn(NN *nn, ELEMENT_TYPE eps, ELEMENT_TYPE learning_rate, size_t learning_iterations, Mat *training_input, Mat *training_output)
{
    size_t arch_count = nn->arch_count;
    for (size_t i = 0; i < learning_iterations; i++)
    {
        ELEMENT_TYPE c = cost_NN(nn, training_input, training_output);
        for (size_t j = 1; j < arch_count; j++)
        {
            diff(nn, training_input, training_output, eps, learning_rate, nn->ws[j], c);
            diff(nn, training_input, training_output, eps, learning_rate, nn->bs[j], c);
        }
        printf("%f\n", c);
    }
}

ELEMENT_TYPE td[] = {
    0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1};

size_t num = sizeof(td) / sizeof(td[0]);

int main()
{

    size_t arch[] = {2, 2, 1};
    size_t stride = 3;
    Mat *t1 = &(Mat){.rows = 4, .cols = 2, .stride = stride, .es = td}; // stored on the function call stack
    Mat *t2 = &(Mat){.rows = 4, .cols = 1, .stride = stride, .es = &td[2]};

    NN *nn = nn_alloc(arch, 3);
    randomize_parameters_NN(nn);

    learn(nn, 1E-1, 1E-1, 100000, t1, t2);

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            ELEMENT_TYPE arr[] = {i, j};
            memcpy(nn->as[0]->es, arr, 2 * sizeof(ELEMENT_TYPE)); // Copy data instead of direct assignment
            forward_NN(nn);

            printf("%zu XOR %zu is %f\n", i, j, nn->as[nn->arch_count - 1]->es[0]);
        }
    }

    return EXIT_SUCCESS;
}
 */

#define NN_IMPLEMENTATION_
#define MODEL GRAD_DESC
#define ELEMENT_TYPE float
#define SIZE_OF_TRAINING_DATA 4
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "nn.h"


ELEMENT_TYPE td[] = {
   /*  0, 0, 0,
    1, 0, 1,
    0, 1, 1,
    1, 1, 0,  */
    // A1, A0, B1, B0, S1, S0
    0, 0, 0, 0, 0, 0, 0, // 00 + 00 = 00
    0, 0, 0, 1, 0, 0, 1, // 00 + 01 = 01
    0, 0, 1, 0, 0, 1, 0, // 00 + 10 = 10
    0, 0, 1, 1, 0, 1, 1, // 00 + 11 = 11
    0, 1, 0, 0, 0, 0, 1, // 01 + 00 = 01
    0, 1, 0, 1, 0, 1, 0, // 01 + 01 = 10
    0, 1, 1, 0, 0, 1, 1, // 01 + 10 = 11
    0, 1, 1, 1, 1, 0, 0, // 01 + 11 = 00 (no carry considered, just the lower 2 bits)
    1, 0, 0, 0, 0, 1, 0, // 10 + 00 = 10
    1, 0, 0, 1, 0, 1, 1, // 10 + 01 = 11
    1, 0, 1, 0, 1, 0, 0, // 10 + 10 = 00 (no carry considered)
    1, 0, 1, 1, 0, 0, 1, // 10 + 11 = 01
    1, 1, 0, 0, 0, 1, 1, // 11 + 00 = 11
    1, 1, 0, 1, 1, 0, 0, // 11 + 01 = 00 (no carry considered)
    1, 1, 1, 0, 0, 0, 1, // 11 + 10 = 01
    1, 1, 1, 1, 1, 1, 0  // 11 + 11 = 10 (no carry considered) 
};

int main()
{
    //omp_set_num_threads(4);
    size_t arch[] = {4, 4, 3};                                       // Architecture of the network
    Mat *t1 = &(Mat){.rows = 16, .cols = 4, .stride = 7, .es = td};     // Input data
    Mat *t2 = &(Mat){.rows = 16, .cols = 3, .stride = 7, .es = &td[4]}; // Output data

    NN *nn = nn_alloc(arch, ARRAY_LEN(arch));
    randomize_parameters_NN(nn, 0, 0);

    NN_PRINT(nn);

    learn(nn, 1e-1, 1e-1, 1000000, t1, t2); // Use smaller eps and learning rate

    NN_PRINT(nn);
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            for (size_t r = 0; r < 2; r++)
            {
                for (size_t k = 0; k < 2; k++)
                {
                    Mat *mat = &(Mat){.rows = 1, .cols = 4, .es = (ELEMENT_TYPE[]){(ELEMENT_TYPE)i, (ELEMENT_TYPE)(j), r, k}};
                    NN_INPUT(nn) = mat;
                    forward_NN(nn);

                    printf("%zu %zu + %zu %zu is %f %f %f\n", i, j, r, k, NN_OUTPUT(nn)->es[0], NN_OUTPUT(nn)->es[1], NN_OUTPUT(nn)->es[2]); 
                }
            } 

           /* Mat *mat = &(Mat){.rows = 1, .cols = 2, .es = (ELEMENT_TYPE[]) {i, j}};
           NN_INPUT(nn) = mat;
           forward_NN(nn);

           printf("%zu XOR %zu is %f\n", i, j, NN_OUTPUT(nn)->es[0]); */
        }
    }

    return EXIT_SUCCESS;
}
