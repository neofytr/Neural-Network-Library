#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>

#define NN_IMPLEMENTATION_
#include "./nn.h" 

#define WINDOW_WIDTH 1200
#define WINDOW_HEIGHT 800
#define NEURON_RADIUS 20
#define LAYER_SPACING 200
#define VERTICAL_SPACING 80

#define COST_WINDOW_WIDTH 800
#define COST_WINDOW_HEIGHT 400
#define MAX_COST_HISTORY 1000
#define COST_GRAPH_PADDING 40

typedef struct
{
    SDL_Window *window;
    SDL_Renderer *renderer;
    float *cost_history;
    int cost_count;
    float min_cost;
    float max_cost;
} CostVisualizer;

typedef struct
{
    SDL_Window *window;
    SDL_Renderer *renderer;
    int running;
    CostVisualizer *cost_vis;
} Visualizer;

void weight_to_color(float weight, Uint8 *r, Uint8 *g, Uint8 *b)
{
    weight = weight < 0 ? 0 : (weight > 1 ? 1 : weight);

    *r = (Uint8)(255 * weight);
    *g = (Uint8)(255 * (1 - weight) + 192 * weight);
    *b = (Uint8)(203 * weight);
}

Visualizer *init_visualizer()
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        fprintf(stderr, "SDL initialization failed: %s\n", SDL_GetError());
        return NULL;
    }

    Visualizer *vis = malloc(sizeof(Visualizer));
    vis->window = SDL_CreateWindow("Neural Network Visualizer",
                                   SDL_WINDOWPOS_UNDEFINED,
                                   SDL_WINDOWPOS_UNDEFINED,
                                   WINDOW_WIDTH, WINDOW_HEIGHT,
                                   SDL_WINDOW_SHOWN);

    if (!vis->window)
    {
        fprintf(stderr, "Window creation failed: %s\n", SDL_GetError());
        free(vis);
        return NULL;
    }

    vis->renderer = SDL_CreateRenderer(vis->window, -1,
                                       SDL_RENDERER_ACCELERATED |
                                           SDL_RENDERER_PRESENTVSYNC);

    if (!vis->renderer)
    {
        fprintf(stderr, "Renderer creation failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(vis->window);
        free(vis);
        return NULL;
    }

    vis->running = 1;
    return vis;
}

CostVisualizer *init_cost_visualizer()
{
    CostVisualizer *cv = malloc(sizeof(CostVisualizer));
    cv->window = SDL_CreateWindow("Training Cost",
                                  SDL_WINDOWPOS_UNDEFINED,
                                  SDL_WINDOWPOS_UNDEFINED,
                                  COST_WINDOW_WIDTH, COST_WINDOW_HEIGHT,
                                  SDL_WINDOW_SHOWN);

    cv->renderer = SDL_CreateRenderer(cv->window, -1,
                                      SDL_RENDERER_ACCELERATED |
                                          SDL_RENDERER_PRESENTVSYNC);

    cv->cost_history = malloc(sizeof(float) * MAX_COST_HISTORY);
    cv->cost_count = 0;
    cv->min_cost = INFINITY;
    cv->max_cost = -INFINITY;

    return cv;
}

void update_cost_graph(CostVisualizer *cv, float cost, long long int iterations)
{
    if (cv->cost_count < MAX_COST_HISTORY)
    {
        cv->cost_history[cv->cost_count++] = cost;
    }
    else
    {
        memmove(cv->cost_history, cv->cost_history + 1,
                (MAX_COST_HISTORY - 1) * sizeof(float));
        cv->cost_history[MAX_COST_HISTORY - 1] = cost;
    }

    cv->min_cost = cost < cv->min_cost ? cost : cv->min_cost;
    cv->max_cost = cost > cv->max_cost ? cost : cv->max_cost;

    SDL_SetRenderDrawColor(cv->renderer, 255, 255, 255, 255);
    SDL_RenderClear(cv->renderer);

    SDL_SetRenderDrawColor(cv->renderer, 0, 0, 0, 255);
    SDL_RenderDrawLine(cv->renderer,
                       COST_GRAPH_PADDING, COST_GRAPH_PADDING,
                       COST_GRAPH_PADDING, COST_WINDOW_HEIGHT - COST_GRAPH_PADDING);
    SDL_RenderDrawLine(cv->renderer,
                       COST_GRAPH_PADDING, COST_WINDOW_HEIGHT - COST_GRAPH_PADDING,
                       COST_WINDOW_WIDTH - COST_GRAPH_PADDING,
                       COST_WINDOW_HEIGHT - COST_GRAPH_PADDING);

    SDL_SetRenderDrawColor(cv->renderer, 255, 0, 0, 255);
    for (int i = 1; i < cv->cost_count; i++)
    {
        float x1 = COST_GRAPH_PADDING +
                   ((i - 1) * (COST_WINDOW_WIDTH - 2 * COST_GRAPH_PADDING)) /
                       (float)MAX_COST_HISTORY;
        float x2 = COST_GRAPH_PADDING +
                   (i * (COST_WINDOW_WIDTH - 2 * COST_GRAPH_PADDING)) /
                       (float)MAX_COST_HISTORY;

        float y1 = COST_WINDOW_HEIGHT - COST_GRAPH_PADDING -
                   ((cv->cost_history[i - 1] - cv->min_cost) *
                    (COST_WINDOW_HEIGHT - 2 * COST_GRAPH_PADDING)) /
                       (cv->max_cost - cv->min_cost);
        float y2 = COST_WINDOW_HEIGHT - COST_GRAPH_PADDING -
                   ((cv->cost_history[i] - cv->min_cost) *
                    (COST_WINDOW_HEIGHT - 2 * COST_GRAPH_PADDING)) /
                       (cv->max_cost - cv->min_cost);

        SDL_RenderDrawLineF(cv->renderer, x1, y1, x2, y2);
    }

   /*  char stats_text[64];
    snprintf(stats_text, sizeof(stats_text), "Iteration: %zu", iterations);
    stringRGBA(cv->renderer, 10, 10, stats_text, 0, 0, 0, 255);

    snprintf(stats_text, sizeof(stats_text), "Cost: %.6f", cost);
    stringRGBA(cv->renderer, 10, 30, stats_text, 0, 0, 0, 255); */

    SDL_RenderPresent(cv->renderer);
}

void draw_network(Visualizer *vis, NN *nn)
{
    SDL_SetRenderDrawColor(vis->renderer, 255, 255, 255, 255);
    SDL_RenderClear(vis->renderer);

    int total_width = (nn->arch_count - 1) * LAYER_SPACING;
    int start_x = (WINDOW_WIDTH - total_width) / 2;

    for (size_t layer = 1; layer < nn->arch_count; layer++)
    {
        int x1 = start_x + (layer - 1) * LAYER_SPACING;
        int x2 = start_x + layer * LAYER_SPACING;

        for (size_t i = 0; i < nn->arch[layer - 1]; i++)
        {
            int y1 = WINDOW_HEIGHT / 2 + (i - nn->arch[layer - 1] / 2.0) * VERTICAL_SPACING;

            for (size_t j = 0; j < nn->arch[layer]; j++)
            {
                int y2 = WINDOW_HEIGHT / 2 + (j - nn->arch[layer] / 2.0) * VERTICAL_SPACING;

                float weight = MAT_AT(nn->ws[layer], i, j);
                Uint8 r, g, b;
                weight_to_color(weight, &r, &g, &b);

                lineRGBA(vis->renderer, x1, y1, x2, y2, r, g, b, 255);
            }
        }
    }

    for (size_t layer = 0; layer < nn->arch_count; layer++)
    {
        int x = start_x + layer * LAYER_SPACING;

        for (size_t i = 0; i < nn->arch[layer]; i++)
        {
            int y = WINDOW_HEIGHT / 2 + (i - nn->arch[layer] / 2.0) * VERTICAL_SPACING;

            filledCircleRGBA(vis->renderer, x, y, NEURON_RADIUS,
                             100, 100, 100, 255);

            circleRGBA(vis->renderer, x, y, NEURON_RADIUS,
                       0, 0, 0, 255);

            if (nn->as[layer] != NULL)
            {
                char activation_text[32];
                snprintf(activation_text, sizeof(activation_text), "%.2f",
                         MAT_AT(nn->as[layer], 0, i));
                stringRGBA(vis->renderer, x - 15, y - 6, activation_text,
                           0, 0, 0, 255);
            }
        }
    }

    SDL_RenderPresent(vis->renderer);
}

void cleanup_visualizer(Visualizer *vis)
{
    if (vis)
    {
        if (vis->cost_vis)
        {
            SDL_DestroyRenderer(vis->cost_vis->renderer);
            SDL_DestroyWindow(vis->cost_vis->window);
            free(vis->cost_vis->cost_history);
            free(vis->cost_vis);
        }
        SDL_DestroyRenderer(vis->renderer);
        SDL_DestroyWindow(vis->window);
        free(vis);
    }
    SDL_Quit();
}

void learn_with_visualization(NN *nn, ELEMENT_TYPE eps, ELEMENT_TYPE learning_rate,
                              size_t learning_iterations, Mat *training_input,
                              Mat *training_output, Visualizer *vis)
{
    for (size_t i = 0; i < learning_iterations && vis->running; i++)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                vis->running = 0;
                return;
            }
        }

#if MODEL == DIFF
        size_t arch_count = nn->arch_count;
        ELEMENT_TYPE c = cost_NN(nn, training_input, training_output);
        for (size_t j = 1; j < arch_count; j++)
        {
            diff(nn, training_input, training_output, eps, learning_rate, nn->ws[j], c);
            diff(nn, training_input, training_output, eps, learning_rate, nn->bs[j], c);
        }
#elif MODEL == GRAD_DESC
        delta(nn, training_input, training_output, learning_rate);
#endif

        ELEMENT_TYPE current_cost = cost_NN(nn, training_input, training_output);
        update_cost_graph(vis->cost_vis, current_cost, i);

        draw_network(vis, nn);
        // SDL_Delay(16); // Cap at ~60 FPS
    }

    ELEMENT_TYPE cost = cost_NN(nn, training_input, training_output);
    printf("Final Cost: %f\n", cost);
}
