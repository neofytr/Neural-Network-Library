#include "./nn_visualizer.h"

ELEMENT_TYPE xor_data[] = {
    // Input pairs | Expected output
    1, 0, 0,
    0, 1, 1,
    1, 1, 0,
    0, 0, 0};

int main(void)
{
    size_t arch[] = {2, 2, 1};

    Mat *training_input = &(Mat){
        .rows = 4,
        .cols = 2,
        .stride = 3,
        .es = &xor_data[0]};

    Mat *training_output = &(Mat){
        .rows = 4,
        .cols = 1,
        .stride = 3,
        .es = &xor_data[2]};

    NN *nn = nn_alloc(arch, ARRAY_LEN(arch));
    if (!nn)
    {
        fprintf(stderr, "Failed to allocate neural network\n");
        return 1;
    }

    randomize_parameters_NN(nn, 0, 0);

    Visualizer *vis = init_visualizer();
    if (!vis)
    {
        fprintf(stderr, "Failed to initialize visualizer\n");
        // nn_free(nn);
        return 1;
    }

    printf("Training XOR gate neural network...\n");
    printf("Press ESC to stop training early\n");

    learn_with_visualization(
        nn,
        1e-2,
        1e-2,
        10000,
        training_input,
        training_output,
        vis);

    printf("\nFinal XOR predictions:\n");
    for (size_t i = 0; i < training_input->rows; i++)
    {
        ELEMENT_TYPE input1 = MAT_AT(training_input, i, 0);
        ELEMENT_TYPE input2 = MAT_AT(training_input, i, 1);

        /* forward_NN(nn, &(Mat){
                           .rows = 1,
                           .cols = 2,
                           .stride = 2,
                           .es = (ELEMENT_TYPE[]){input1, input2}}); */

        ELEMENT_TYPE prediction = MAT_AT(nn->as[nn->arch_count - 1], 0, 0);
        ELEMENT_TYPE expected = MAT_AT(training_output, i, 0);

        printf("Input: %g XOR %g = %g (Expected: %g)\n",
               input1, input2, prediction, expected);
    }

    cleanup_visualizer(vis);

    return 0;
}