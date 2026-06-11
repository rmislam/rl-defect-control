#include "cuda_params.h"
#include "parameters.c"

dim3 BLOCK_DIM(BLOCK_SIZE, BLOCK_SIZE);
dim3 GRID_DIM((I + BLOCK_SIZE - 1) / BLOCK_SIZE, (J + BLOCK_SIZE - 1) / BLOCK_SIZE);