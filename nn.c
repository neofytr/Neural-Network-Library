#define NN_IMPLEMENTATION_
#define ELEMENT_TYPE double
#define ARRAY_LEN(xs) (sizeof(xs) / sizeof(xs[0]))
#define SIZE_OF_TRAINING_DATA 10

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
        mat_rand(nn->ws[i], 0, 0);
        mat_rand(nn->bs[i], 0, 0);
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
    assert(training_input->rows == training_output->rows);
    assert(nn->as[nn->arch_count - 1]->cols == training_output->cols);

    size_t rows = training_input->rows;
    size_t cols = training_output->cols;

    ELEMENT_TYPE result = (ELEMENT_TYPE)0;
    for (size_t i = 0; i < rows; i++)
    {
        Mat NN_input = mat_row(training_input, i);
        *nn->as[0] = NN_input;
        forward_NN(nn);

        Mat NN_output = *nn->as[nn->arch_count - 1];
        Mat *NN_output_ = &NN_output;
        for (size_t j = 0; j < cols; j++)
        {
            ELEMENT_TYPE y = MAT_AT(NN_output_, 0, j);
            ELEMENT_TYPE d = y - MAT_AT(training_output, 0, j);
            result += d * d;
        }
    }

    return (result / rows);
}

void diff(NN *nn, Mat *training_input, Mat *training_output, size_t eps, size_t learning_rate, Mat *temp_para)
{
    ELEMENT_TYPE saved;
    size_t rows = temp_para->rows, cols = temp_para->cols;
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            saved = MAT_AT(temp_para, i, j);
            MAT_AT(temp_para, i, j) += eps;
            ELEMENT_TYPE d = (cost_NN(nn, training_input, training_output) - c) / eps;
            MAT_AT(temp_para, i, j) = saved;
            MAT_AT(temp_para, i, j) -= d * learning_rate;
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
            diff(nn, training_input, training_output, eps, learning_rate, nn->ws[i], c);
            diff(nn, training_input, training_output, eps, learning_rate, nn->bs[i]);
            diff(nn, training_input, training_output, eps, learning_rate, nn->as[i]);
        }
    }
}

ELEMENT_TYPE td[] = {
    0,
    0,
    0,
    1,
    0,
    1,
    0,
    1,
    1,
    1,
    1,
    0,
};

size_t num = sizeof(td) / sizeof(td[0]);

typedef struct
{
    Mat *a0;
    Mat *w1, *b1, *a1;
    Mat *w2, *b2, *a2;
} Xor;

Xor xor_alloc()
{
    Xor m;

    m.a0 = mat_alloc(1, 2);

    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2);
    m.a1 = mat_alloc(1, 2);

    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);
    m.a2 = mat_alloc(1, 1);

    return m;
}

void forward_xor(Xor *m)
{
    mat_dot(m->a1, m->a0, m->w1);
    mat_sum(m->a1, m->b1);
    mat_sig(m->a1);

    mat_dot(m->a2, m->a1, m->w2);
    mat_sum(m->a2, m->b2);
    mat_sig(m->a2);
}

ELEMENT_TYPE cost(Xor *m, Mat *ti, Mat *to)
{
    assert(ti->rows == to->rows);
    assert(to->cols == m->a2->cols);
    size_t n = ti->rows;
    size_t r = to->cols;

    ELEMENT_TYPE result = (ELEMENT_TYPE)0;
    for (size_t i = 0; i < n; i++)
    {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);
        Mat *y_ = &y;

        *(m->a0) = x;
        forward_xor(m);

        for (size_t j = 0; j < r; j++)
        {
            ELEMENT_TYPE d = MAT_AT(m->a2, 0, j) - MAT_AT(y_, 0, j);
            result += d * d;
        }
    }

    return (result / n);
}

void finite_diff(Xor *m, Xor *g, ELEMENT_TYPE eps, Mat *ti, Mat *to)
{
    ELEMENT_TYPE saved;
    ELEMENT_TYPE c = cost(m, ti, to);

    for (size_t i = 0; i < m->w1->rows; i++)
    {
        for (size_t j = 0; j < m->w1->cols; j++)
        {
            saved = MAT_AT(m->w1, i, j);
            *MAT_AT_POINTER(m->w1, i, j) += eps;
            *MAT_AT_POINTER(g->w1, i, j) = (cost(m, ti, to) - c) / eps;
            *MAT_AT_POINTER(m->w1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m->w2->rows; i++)
    {
        for (size_t j = 0; j < m->w2->cols; j++)
        {
            saved = MAT_AT(m->w2, i, j);
            *MAT_AT_POINTER(m->w2, i, j) += eps;
            *MAT_AT_POINTER(g->w2, i, j) = (cost(m, ti, to) - c) / eps;
            *MAT_AT_POINTER(m->w2, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m->b1->rows; i++)
    {
        for (size_t j = 0; j < m->b1->cols; j++)
        {
            saved = MAT_AT(m->b1, i, j);
            *MAT_AT_POINTER(m->b1, i, j) += eps;
            *MAT_AT_POINTER(g->b1, i, j) = (cost(m, ti, to) - c) / eps;
            *MAT_AT_POINTER(m->b1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m->b2->rows; i++)
    {
        for (size_t j = 0; j < m->b2->cols; j++)
        {
            saved = MAT_AT(m->b2, i, j);
            *MAT_AT_POINTER(m->b2, i, j) += eps;
            *MAT_AT_POINTER(g->b2, i, j) = (cost(m, ti, to) - c) / eps;
            *MAT_AT_POINTER(m->b2, i, j) = saved;
        }
    }
}

void xor_learn(Xor *m, Xor *g, ELEMENT_TYPE learning_rate)
{
    for (size_t i = 0; i < m->w1->rows; i++)
    {
        for (size_t j = 0; j < m->w1->cols; j++)
        {
            *MAT_AT_POINTER(m->w1, i, j) -= learning_rate * (MAT_AT(g->w1, i, j));
        }
    }

    for (size_t i = 0; i < m->w2->rows; i++)
    {
        for (size_t j = 0; j < m->w2->cols; j++)
        {
            *MAT_AT_POINTER(m->w2, i, j) -= learning_rate * (MAT_AT(g->w2, i, j));
        }
    }

    for (size_t i = 0; i < m->b1->rows; i++)
    {
        for (size_t j = 0; j < m->b1->cols; j++)
        {
            *MAT_AT_POINTER(m->b1, i, j) -= learning_rate * (MAT_AT(g->b1, i, j));
        }
    }

    for (size_t i = 0; i < m->b2->rows; i++)
    {
        for (size_t j = 0; j < m->b2->cols; j++)
        {
            *MAT_AT_POINTER(m->b2, i, j) -= learning_rate * (MAT_AT(g->b2, i, j));
        }
    }
}

int main()
{
    Xor m = xor_alloc();
    Xor g = xor_alloc();

    mat_rand(m.w1, 1, 0);
    mat_rand(m.w2, 1, 0);
    mat_rand(m.b1, 1, 0);
    mat_rand(m.b2, 1, 0);

    size_t stride = 3;
    Mat *t1 = &(Mat){.rows = num / stride, .cols = 2, .stride = stride, .es = td}; // stored on the function call stack
    Mat *t2 = &(Mat){.rows = num / stride, .cols = 1, .stride = stride, .es = &td[2]};

    printf("cost: %f\n", cost(&m, t1, t2));
    for (size_t i = 0; i < 100000; i++)
    {
        finite_diff(&m, &g, 1e-1, t1, t2);
        xor_learn(&m, &g, 1e-1);
        printf("%zu: cost: %f\n", i, cost(&m, t1, t2));
    }

    /* MAT_PRINT(t1);
    MAT_PRINT(t2); */

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            ELEMENT_TYPE arr[] = {i, j};
            (m.a0->es) = arr;
            forward_xor(&m);
            printf("%zu XOR %zu is %f\n", i, j, m.a2->es[0]);
        }
    }

    return EXIT_SUCCESS;
}