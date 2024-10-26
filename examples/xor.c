#include "../nn_visualizer.h"

ELEMENT_TYPE xor [] = {
    1,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
    0,
    0,
    1,
};

int main()
{
    size_t arch[] = {2, 2, 1}; // 2 inputs, 2 hidden layers, 1 output

    Mat *training_input = &(Mat){.rows = 4, .cols = 2, .stride = 3, .es = &xor[0]};
    Mat *training_output = &(Mat){.rows = 4, .cols = 1, .stride = 3, .es = &xor[2]};

    NN *nn = nn_alloc(arch, ARRAY_LEN(arch));
    randomize_parameters_NN(nn, 0, 0); // Initialize weights/biases

    Visualizer *vis = init_visualizer();
    if (!vis)
    {
        fprintf(stderr, "Failed to initialize visualizer\n");
        return 1;
    }

    vis->cost_vis = init_cost_visualizer();
    if (!vis->cost_vis)
    {
        fprintf(stderr, "Failed to initialize cost visualizer\n");
        cleanup_visualizer(vis);
        return 1;
    }

    // Train network with visualization
    learn_with_visualization(nn, 1e-3, 0.1, 100000,
                             training_input, training_output, vis);

    // Cleanup
    cleanup_visualizer(vis);

    return 0;
}