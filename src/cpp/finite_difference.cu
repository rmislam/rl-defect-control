#include "cuda_params.h"

//Make a step in the Q-tensor finite difference evolution
void compute_FD_step(float* __restrict__ U, float* __restrict__ Q, float* __restrict__ QNEW, float* __restrict__ H, char* __restrict__ LMARK) {
    computeQNEW<<<GRID_DIM, BLOCK_DIM>>>(U, Q, QNEW, H, LMARK);
    calcQNEW2Q<<<GRID_DIM, BLOCK_DIM>>>(Q, QNEW, LMARK);
}

// This function modifies QNEW and H, but leaves U, Q, and LMARK unmodified
__global__
void computeQNEW(float* __restrict__ U, float* __restrict__ Q, float* __restrict__ QNEW, float* __restrict__ H, char* __restrict__ LMARK) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= I || y >= J) return;
    int l = y * I + x;

    if (LMARK[l] == LMARK_BULK) {
        float u1[2], u2[2], u3[2];
        //Elastic part
        float Q_laplacian[2];

        // compute_Q_laplacian
        int lxp = l + 1;    //Indices of neighbouring mesh points
        int lxm = l - 1;
        int lyp = l + I;
        int lym = l - I;
        
        // Laplacian of tensor is simply the Laplacian of each component
        #pragma unroll
        for (int m = 0; m < 2; m++) {
            // five-point stencil finite-difference
            Q_laplacian[m] = Q[m * NMAX + lxp] + Q[m * NMAX + lxm] + Q[m * NMAX + lyp] + Q[m * NMAX + lym] - 4 * Q[m * NMAX + l];
        }

        // compute_u1
        float Q0 = Q[l];
        float Q1 = Q[NMAX + l];
        float Qsq00 = Q0 * Q0 + Q1 * Q1;

        u1[0] = L * Q_laplacian[0] - A * Q0 - 2 * C * Q0 * Qsq00;
        u1[1] = L * Q_laplacian[1] - A * Q1 - 2 * C * Q1 * Qsq00;

        //Write u1 as H in a global matrix
        // Remember H is molecular field
        H[l] = u1[0];
        H[NMAX + l] = u1[1];

        // compute_u2 (Shear part)
        float uxx = (U[NMAX + l + 1] - U[NMAX + l - 1]) / 2.0f;
        float uyy = -uxx;  // Remember that div u = 0 (incompressibility condition), so uxx = -uyy
        float uxy = (U[NMAX + l + I] - U[NMAX + l - I]) / 2.0f;
        float uyx = (U[2 * NMAX + l + 1] - U[2 * NMAX + l - 1]) / 2.0f;

        // u2 is S in the Beris-Edwards equation
        u2[0] = Q1 * (uxy - uyx) + XI * (uxx + 2 * Q0 * Q0 * (uyy - uxx) - 2 * Q0 * Q1 * (uyx + uxy));
        u2[1] = (0.5f * XI - Q0) * uxy + (0.5f * XI + Q0) * uyx;

        // compute_u3 (Advective part)
        // Gradient of Q
        float dQxxdx = (Q[l + 1] - Q[l - 1]) / 2.0f;
        float dQxydx = (Q[NMAX + l + 1] - Q[NMAX + l - 1]) / 2.0f;
        float dQxxdy = (Q[l + I] - Q[l - I]) / 2.0f;
        float dQxydy = (Q[NMAX + l + I] - Q[NMAX + l - I]) / 2.0f;

        float ux = U[NMAX + l];
        float uy = U[2 * NMAX + l];

        // -u dot grad Q
        u3[0] = -(ux * dQxxdx + uy * dQxxdy);
        u3[1] = -(ux * dQxydx + uy * dQxydy);

        //Make the time step
        #pragma unroll
        for (int m = 0; m < 2; m++) {
            QNEW[m * NMAX + l] = Q[m * NMAX + l] + (GAMMA * u1[m] + u2[m] + u3[m]) * DT / (float)(N_EVOL_Q);
        }
    }

    float degree_of_order = 1.;
    float angle = 0.5 * M_PI; // vertical (homeotropic) anchoring at top and bottom boundaries  //0.25 * M_PI;

    if (LMARK[l] == LMARK_TOP_WALL || LMARK[l] == LMARK_BOT_WALL) { // walls
        angle = 0.5 * M_PI;
        QNEW[l] = degree_of_order / 2.0 * cos(2 * angle);   //Qxx component
        QNEW[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);   //Qxy component
    } else if (LMARK[l] == LMARK_LEFT_WALL || LMARK[l] == LMARK_RIGHT_WALL) {
        angle = 0;
        QNEW[l] = degree_of_order / 2.0 * cos(2 * angle);   //Qxx component
        QNEW[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);   //Qxy component
    } else if (LMARK[l] == LMARK_CORNER_TOP_LEFT_25 || LMARK[l] == LMARK_CORNER_BOT_RIGHT_25) { // corners
        angle = 0.25 * M_PI;  // NOTE: careful: this should be 0.25, not 0.75, in order to prevent defects
        QNEW[l] = degree_of_order / 2.0 * cos(2 * angle);   //Qxx component
        QNEW[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);   //Qxy component
    } else if (LMARK[l] == LMARK_CORNER_TOP_RIGHT_25 || LMARK[l] == LMARK_CORNER_BOT_LEFT_25) {
        angle = 0.25 * M_PI;
        QNEW[l] = degree_of_order / 2.0 * cos(2 * angle);   //Qxx component
        QNEW[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);   //Qxy component
    } else if (LMARK[l] == LMARK_CORNER_TOP_LEFT_75 || LMARK[l] == LMARK_CORNER_BOT_RIGHT_75) { // corners
        angle = 0.75 * M_PI;  // NOTE: careful: reverse the direction compared to the above two blocks
        QNEW[l] = degree_of_order / 2.0 * cos(2 * angle);   //Qxx component
        QNEW[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);   //Qxy component
    } else if (LMARK[l] == LMARK_CORNER_TOP_RIGHT_75 || LMARK[l] == LMARK_CORNER_BOT_LEFT_75) {
        angle = 0.75 * M_PI;
        QNEW[l] = degree_of_order / 2.0 * cos(2 * angle);   //Qxx component
        QNEW[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);   //Qxy component
    }
}

// Write QNEW back to Q and implement periodic boundary conditions
// This function modifies Q, but leaves QNEW and LMARK unmodified
__global__
void calcQNEW2Q(float* __restrict__ Q, float* __restrict__ QNEW, char* __restrict__ LMARK) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= I || y >= J) return;
    int l = y * I + x;

    if (LMARK[l] == LMARK_TOP_OUTLET_PBC) {  // outlets -- periodic boundaries
        #pragma unroll
        for(int m = 0; m < 2; m++) {
            Q[m * NMAX + l] = QNEW[m * NMAX + l - I * (J - 2)];
        }
    } else if (LMARK[l] == LMARK_BOT_OUTLET_PBC) {
        #pragma unroll
        for(int m = 0; m < 2; m++) {
            Q[m * NMAX + l] = QNEW[m * NMAX + l + I * (J - 2)];
        }
    } else if (LMARK[l] == LMARK_LEFT_OUTLET_PBC) {
        #pragma unroll
        for(int m = 0; m < 2; m++) {
            Q[m * NMAX + l] = QNEW[m * NMAX + l + I - 2];
        }
    } else if (LMARK[l] == LMARK_RIGHT_OUTLET_PBC) {
        #pragma unroll
        for(int m = 0; m < 2; m++) {
            Q[m * NMAX + l] = QNEW[m * NMAX + l - (I - 2)];
        }
    } else if (LMARK[l] == LMARK_TOP_OUTLET_OBC) {  // outlets -- open boundaries
        #pragma unroll
        for(int m = 0; m < 2; m++) {
            Q[m * NMAX + l] = QNEW[m * NMAX + l - I];
        }
    } else if (LMARK[l] == LMARK_BOT_OUTLET_OBC) {
        #pragma unroll
        for(int m = 0; m < 2; m++) {
            Q[m * NMAX + l] = QNEW[m * NMAX + l + I];
        }
    } else if (LMARK[l] == LMARK_LEFT_OUTLET_OBC) {
        #pragma unroll
        for(int m = 0; m < 2; m++) {
            Q[m * NMAX + l] = QNEW[m * NMAX + l + 1];
        }
    } else if (LMARK[l] == LMARK_RIGHT_OUTLET_OBC) {
        #pragma unroll
        for(int m = 0; m < 2; m++) {
            Q[m * NMAX + l] = QNEW[m * NMAX + l - 1];
        }
    } else {
        #pragma unroll
        for(int m = 0; m < 2; m++) {
            Q[m * NMAX + l] = QNEW[m * NMAX + l];
        }
    }
}
