#include "cuda_params.h"

//Characteristic velocity vectors of the LB model.

//#################################
//#  D2Q9 Structure
//#################################
//#  6     2     5        y
//#    .   .   .          ^
//#      . . .            |
//#  3 . . 0 . . 1        ----> x
//#      . . .
//#    .   .   .
//#  7     4     8
//#################################

__constant__ float WKONST[LATTICE_VELOCITY_NUMBER];
__constant__ float E[2 * LATTICE_VELOCITY_NUMBER];

const float h_WKONST[LATTICE_VELOCITY_NUMBER] = { 4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 36., 1. / 36., 1. / 36., 1. / 36. };
const float h_E[2 * LATTICE_VELOCITY_NUMBER] = {
    0, 1, 0, -1,  0, 1, -1, -1,  1,
    0, 0, 1,  0, -1, 1,  1, -1, -1
};

void initCUDAConstants() {
    // Initialize CUDA constants
    cudaMemcpyToSymbol(WKONST, h_WKONST, sizeof(h_WKONST));
    cudaMemcpyToSymbol(E, h_E, sizeof(h_E));
}

//Makes a main step in the LB part
void compute_LB_step(float* __restrict__ U, float* __restrict__ FNEW, float* __restrict__ F, float* __restrict__ FEQ, float* __restrict__ P, float* __restrict__ Q, float* __restrict__ H, float* __restrict__ SIGMA, float* __restrict__ ACTIVITY, char* __restrict__ LMARK) {
    computeFeq<<<GRID_DIM, BLOCK_DIM>>>(U, FEQ);   //Compute the equilibrium distribution
    computeSigma<<<GRID_DIM, BLOCK_DIM>>>(SIGMA, Q, H, ACTIVITY, LMARK);  //Compute stress tensor
    computeP<<<GRID_DIM, BLOCK_DIM>>>(U, P, SIGMA, LMARK);  //Compute the forcing terms
    computeFNEW<<<GRID_DIM, BLOCK_DIM>>>(FNEW, F, FEQ, P);  // Compute the new distribution values (i.e., apply the Boltzmann equation)
    enforceBCAndCalcFNEW2F2U<<<GRID_DIM, BLOCK_DIM>>>(FNEW, F, U, SIGMA, LMARK);
}

//Computes equilibrium distribution function
__global__
void computeFeq(float* __restrict__ U, float* __restrict__ FEQ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= I || y >= J) return;
    int l = y * I + x;

    float rho = U[l];
    float ux = U[NMAX + l];
    float uy = U[2 * NMAX + l];
    float u2 = ux * ux + uy * uy;	//Velocity squared

    #pragma unroll
    for (int m = 0; m < LATTICE_VELOCITY_NUMBER; m++) {
        float ex = E[m];
        float ey = E[m + LATTICE_VELOCITY_NUMBER];
        float ue = ux * ex + uy * ey;		//Product of velocity and characteristic vectors
        FEQ[m * NMAX + l] = WKONST[m] * rho * (1.0f + 3.0f * ue + 9.0f / 2.0f * ue * ue - 1.5f * u2);  // NOTE: no need to modify
    }
}

//Compute stress tensor
__global__
void computeSigma(float* __restrict__ SIGMA, float* __restrict__ Q, float* __restrict__ H, float* __restrict__ ACTIVITY, char* __restrict__ LMARK) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= I || y >= J) return;
    int l = y * I + x;

    if (LMARK[l] != LMARK_BULK) return;
    // NOTE: positive activity is extensile, negative activity is contractile
    // NOTE: Qxx = Q[0], Qyy = -Q[0], Qxy = Qyx = Q[1], and Hxx = H[0], Hyy = -H[0], Hxy = Hyx = H[1]
    float Q0_xp = Q[l + 1];
    float Q0_xm = Q[l - 1];
    float Q1_xp = Q[NMAX + l + 1];
    float Q1_xm = Q[NMAX + l - 1];
    float Q0_yp = Q[l + I];
    float Q0_ym = Q[l - I];
    float Q1_yp = Q[NMAX + l + I];
    float Q1_ym = Q[NMAX + l - I];

    float dxQ0 = (Q0_xp - Q0_xm) / 2.0f;
    float dxQ1 = (Q1_xp - Q1_xm) / 2.0f;
    float dyQ0 = (Q0_yp - Q0_ym) / 2.0f;
    float dyQ1 = (Q1_yp - Q1_ym) / 2.0f;
    float gradQsq = dxQ0 * dxQ0 + dxQ1 * dxQ1 + dyQ0 * dyQ0 + dyQ1 * dyQ1;

    float act = ACTIVITY[l];
    float Q0 = Q[l];
    float Q1 = Q[NMAX + l];
    float H0 = H[l];
    float H1 = H[NMAX + l];

    //xx component of the stress tensor
    SIGMA[l] = L * gradQsq + 4 * XI * (Q0 * H0 + Q1 * H1) * (Q0 + 0.5f) - XI * (2 * (Q0 * H0 + Q1 * H1) + H0) - 2 * L * (dxQ0 * dxQ0 + dxQ1 * dxQ1) - act * Q0;
    //xy component of the stress tensor
    SIGMA[NMAX + l] = 4 * XI * (Q0 * H0 + Q1 * H1) * Q1 - XI * H1 - 2 * L * (dxQ0 * dyQ0 + dxQ1 * dyQ1) + 2 * (Q0 * H1 - Q1 * H0) - act * Q1;
    //yx component of the stress tensor
    SIGMA[2 * NMAX + l] = 4 * XI * (Q0 * H0 + Q1 * H1) * Q1 - XI * H1 - 2 * L * (dxQ0 * dyQ0 + dxQ1 * dyQ1) + 2 * (Q1 * H0 - Q0 * H1) - act * Q1;
    //yy component of the stress tensor
    SIGMA[3 * NMAX + l] = L * gradQsq + 4 * XI * (Q0 * H0 + Q1 * H1) * (-Q0 + 0.5f) - XI * (2 * (Q0 * H0 + Q1 * H1) - H0) - 2 * L * (dyQ0 * dyQ0 + dyQ1 * dyQ1) + act * Q0;
}

//Compute the forcing terms
__global__
void computeP(float* __restrict__ U, float* __restrict__ P, float* __restrict__ SIGMA, char* __restrict__ LMARK) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= I || y >= J) return;
    int l = y * I + x;
    if (LMARK[l] != LMARK_BULK) return;

    float ux = U[NMAX + l];
    float uy = U[2 * NMAX + l];
    float S0_xp = SIGMA[l + 1];
    float S0_xm = SIGMA[l - 1];
    float S1_yp = SIGMA[NMAX + l + I];
    float S1_ym = SIGMA[NMAX + l - I];
    float S2_xp = SIGMA[2 * NMAX + l + 1];
    float S2_xm = SIGMA[2 * NMAX + l - 1];
    float S3_yp = SIGMA[3 * NMAX + l + I];
    float S3_ym = SIGMA[3 * NMAX + l - I];

    //Compute the derivatives of the stress tensor and the force
    float forceX = (S0_xp - S0_xm) / 2.0f + (S1_yp - S1_ym) / 2.0f - MU * ux;
    float forceY = (S2_xp - S2_xm) / 2.0f + (S3_yp - S3_ym) / 2.0f - MU * uy;

    float uF = ux * forceX + uy * forceY;  //Product of force and velocity

    #pragma unroll
    for (int m = 0; m < LATTICE_VELOCITY_NUMBER; m++) {
        float ex = E[m];
        float ey = E[m + LATTICE_VELOCITY_NUMBER];
        float ue = ux * ex + uy * ey;	//Product of velocity and characteristic vectors
        float eF = ex * forceX + ey * forceY;  //Product of force and characteristic vectors
        P[m * NMAX + l] = (1.0f - DT / 2.0f / TAUF) * WKONST[m] * (3.0f * eF - 3.0f * uF + 9.0f * ue * eF);  // "source term" S_i from Kruger 6.2
    }
}

__global__
void computeFNEW(float* __restrict__ FNEW, float* __restrict__ F, float* __restrict__ FEQ, float* __restrict__ P) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= I || y >= J) return;
    int l = y * I + x;

    #pragma unroll
    for (int m = 0; m < LATTICE_VELOCITY_NUMBER; m++) {
        // Compute streaming location
        float ex = E[m];
        float ey = E[m + LATTICE_VELOCITY_NUMBER];
        int lnew = (l % I) + ex + (((l % (I * J)) / I) + ey) * I;  // equivalent to i + j * I
        if (lnew >= NMAX || lnew < 0) continue;

        if (DT == 1 && TAUF == 1) {
            FNEW[m * NMAX + lnew] = FEQ[m * NMAX + l] + P[m * NMAX + l];
        } else {
            FNEW[m * NMAX + lnew] = F[m * NMAX + l] + DT * ((FEQ[m * NMAX + l] - F[m * NMAX + l]) / TAUF + P[m * NMAX + l]);
        }
    }
}

__global__
void enforceBCAndCalcFNEW2F2U(float* __restrict__ FNEW, float* __restrict__ F, float* __restrict__ U, float* __restrict__ SIGMA, char* __restrict__ LMARK) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= I || y >= J) return;
    int l = y * I + x;

    // enforce boundary conditions
    if (LMARK[l] == LMARK_TOP_WALL) {  // walls -- no-slip
        FNEW[4 * NMAX + l] = F[2 * NMAX + l];
        FNEW[7 * NMAX + l] = F[5 * NMAX + l];
        FNEW[8 * NMAX + l] = F[6 * NMAX + l];
    } else if (LMARK[l] == LMARK_BOT_WALL) {
        FNEW[2 * NMAX + l] = F[4 * NMAX + l];
        FNEW[5 * NMAX + l] = F[7 * NMAX + l];
        FNEW[6 * NMAX + l] = F[8 * NMAX + l];
    } else if (LMARK[l] == LMARK_RIGHT_WALL) {
        FNEW[3 * NMAX + l] = F[NMAX + l];
        FNEW[6 * NMAX + l] = F[8 * NMAX + l];
        FNEW[7 * NMAX + l] = F[5 * NMAX + l];
    } else if (LMARK[l] == LMARK_LEFT_WALL) {
        FNEW[NMAX + l] = F[3 * NMAX + l];
        FNEW[8 * NMAX + l] = F[6 * NMAX + l];
        FNEW[5 * NMAX + l] = F[7 * NMAX + l];
    } else if (LMARK[l] == LMARK_CORNER_TOP_LEFT_25 || LMARK[l] == LMARK_CORNER_TOP_LEFT_75) {  // corners -- no-slip
        FNEW[NMAX + l] = F[3 * NMAX + l];
        FNEW[4 * NMAX + l] = F[2 * NMAX + l];
        FNEW[8 * NMAX + l] = F[6 * NMAX + l];
    } else if (LMARK[l] == LMARK_CORNER_BOT_LEFT_25 || LMARK[l] == LMARK_CORNER_BOT_LEFT_75) {
        FNEW[NMAX + l] = F[3 * NMAX + l];
        FNEW[2 * NMAX + l] = F[4 * NMAX + l];
        FNEW[5 * NMAX + l] = F[7 * NMAX + l];
    } else if (LMARK[l] == LMARK_CORNER_TOP_RIGHT_25 || LMARK[l] == LMARK_CORNER_TOP_RIGHT_75) {
        FNEW[3 * NMAX + l] = F[NMAX + l];
        FNEW[4 * NMAX + l] = F[2 * NMAX + l];
        FNEW[7 * NMAX + l] = F[5 * NMAX + l];
    } else if (LMARK[l] == LMARK_CORNER_BOT_RIGHT_25 || LMARK[l] == LMARK_CORNER_BOT_RIGHT_75) {
        FNEW[3 * NMAX + l] = F[NMAX + l];
        FNEW[2 * NMAX + l] = F[4 * NMAX + l];
        FNEW[6 * NMAX + l] = F[8 * NMAX + l];
    } else if (LMARK[l] == LMARK_TOP_OUTLET_PBC) {  // outlets -- periodic boundaries
        FNEW[4 * NMAX + l] = F[4 * NMAX + l - I * (J - 2)];
        FNEW[7 * NMAX + l] = F[7 * NMAX + l - I * (J - 2)];
        FNEW[8 * NMAX + l] = F[8 * NMAX + l - I * (J - 2)];
    } else if (LMARK[l] == LMARK_BOT_OUTLET_PBC) {
        FNEW[2 * NMAX + l] = F[2 * NMAX + l + I * (J - 2)];
        FNEW[5 * NMAX + l] = F[5 * NMAX + l + I * (J - 2)];
        FNEW[6 * NMAX + l] = F[6 * NMAX + l + I * (J - 2)];
    } else if (LMARK[l] == LMARK_LEFT_OUTLET_PBC) {
        FNEW[NMAX + l] = F[NMAX + l + I - 2];
        FNEW[5 * NMAX + l] = F[5 * NMAX + l + I - 2];
        FNEW[8 * NMAX + l] = F[8 * NMAX + l + I - 2];
    } else if (LMARK[l] == LMARK_RIGHT_OUTLET_PBC) {
        FNEW[3 * NMAX + l] = F[3 * NMAX + l - (I - 2)];
        FNEW[6 * NMAX + l] = F[6 * NMAX + l - (I - 2)];
        FNEW[7 * NMAX + l] = F[7 * NMAX + l - (I - 2)];
    } else if (LMARK[l] == LMARK_TOP_OUTLET_OBC) {  // outlets -- open boundaries
        FNEW[4 * NMAX + l] = F[4 * NMAX + l - I];
        FNEW[7 * NMAX + l] = F[7 * NMAX + l - I];
        FNEW[8 * NMAX + l] = F[8 * NMAX + l - I];
    } else if (LMARK[l] == LMARK_BOT_OUTLET_OBC) {
        FNEW[2 * NMAX + l] = F[2 * NMAX + l + I];
        FNEW[5 * NMAX + l] = F[5 * NMAX + l + I];
        FNEW[6 * NMAX + l] = F[6 * NMAX + l + I];
    } else if (LMARK[l] == LMARK_LEFT_OUTLET_OBC) {
        FNEW[NMAX + l] = F[NMAX + l + 1];
        FNEW[5 * NMAX + l] = F[5 * NMAX + l + 1];
        FNEW[8 * NMAX + l] = F[8 * NMAX + l + 1];
    } else if (LMARK[l] == LMARK_RIGHT_OUTLET_OBC) {
        FNEW[3 * NMAX + l] = F[3 * NMAX + l - 1];
        FNEW[6 * NMAX + l] = F[6 * NMAX + l - 1];
        FNEW[7 * NMAX + l] = F[7 * NMAX + l - 1];
    }

    // For lower latency
    float F_local[LATTICE_VELOCITY_NUMBER];

    // Copy FNEW to F
    #pragma unroll
    for (int m = 0; m < LATTICE_VELOCITY_NUMBER; m++) {
        F_local[m] = FNEW[m * NMAX + l];
        F[m * NMAX + l] = F_local[m];
    }

    if (LMARK[l] == LMARK_BULK) {
        float fex = 0.0f, fey = 0.0f, density = 0.0f;

        #pragma unroll
        for (int m = 0; m < LATTICE_VELOCITY_NUMBER; m++) {
            float ex = E[m];
            float ey = E[m + LATTICE_VELOCITY_NUMBER];
            density += F[m * NMAX + l];
            fex += F[m * NMAX + l] * ex;
            fey += F[m * NMAX + l] * ey;
        }

        float ux = U[NMAX + l];
        float uy = U[2 * NMAX + l];
        float S0_xp = SIGMA[l + 1];
        float S0_xm = SIGMA[l - 1];
        float S1_yp = SIGMA[NMAX + l + I];
        float S1_ym = SIGMA[NMAX + l - I];
        float S2_xp = SIGMA[2 * NMAX + l + 1];
        float S2_xm = SIGMA[2 * NMAX + l - 1];
        float S3_yp = SIGMA[3 * NMAX + l + I];
        float S3_ym = SIGMA[3 * NMAX + l - I];

        float forceX = (S0_xp - S0_xm) / 2.0f + (S1_yp - S1_ym) / 2.0f - MU * ux;
        float forceY = (S2_xp - S2_xm) / 2.0f + (S3_yp - S3_ym) / 2.0f - MU * uy;

        U[NMAX + l] = (fex + forceX * DT / 2.0f) / density;
        U[2 * NMAX + l] = (fey + forceY * DT / 2.0f) / density;
        U[l] = density;
    } else if (LMARK[l] == LMARK_TOP_OUTLET_PBC) {  // outlets -- periodic boundaries
        U[NMAX + l] = U[NMAX + l - I * (J - 2)];
        U[2 * NMAX + l] = U[2 * NMAX + l - I * (J - 2)];
    } else if (LMARK[l] == LMARK_BOT_OUTLET_PBC) {
        U[NMAX + l] = U[NMAX + l + I * (J - 2)];
        U[2 * NMAX + l] = U[2 * NMAX + l + I * (J - 2)];
    } else if (LMARK[l] == LMARK_LEFT_OUTLET_PBC) {
        U[NMAX + l] = U[NMAX + l + I - 2];
        U[2 * NMAX + l] = U[2 * NMAX + l + I - 2];
    } else if (LMARK[l] == LMARK_RIGHT_OUTLET_PBC) {
        U[NMAX + l] = U[NMAX + l - (I - 2)];
        U[2 * NMAX + l] = U[2 * NMAX + l - (I - 2)];
    } else if (LMARK[l] == LMARK_TOP_OUTLET_OBC) {  // outlets -- open boundaries
        U[NMAX + l] = U[NMAX + l - I];
        U[2 * NMAX + l] = U[2 * NMAX + l - I];
    } else if (LMARK[l] == LMARK_BOT_OUTLET_OBC) {
        U[NMAX + l] = U[NMAX + l + I];
        U[2 * NMAX + l] = U[2 * NMAX + l + I];
    } else if (LMARK[l] == LMARK_LEFT_OUTLET_OBC) {
        U[NMAX + l] = U[NMAX + l + 1];
        U[2 * NMAX + l] = U[2 * NMAX + l + 1];
    } else if (LMARK[l] == LMARK_RIGHT_OUTLET_OBC) {
        U[NMAX + l] = U[NMAX + l - 1];
        U[2 * NMAX + l] = U[2 * NMAX + l - 1];
    } else if (LMARK[l] == LMARK_TOP_WALL || LMARK[l] == LMARK_BOT_WALL || LMARK[l] == LMARK_LEFT_WALL || LMARK[l] == LMARK_RIGHT_WALL) {  // no-slip condition along channel walls
        U[NMAX + l] = 0;
        U[2 * NMAX + l] = 0;
    } else if (LMARK[l] == LMARK_CORNER_BOT_LEFT_25 || LMARK[l] == LMARK_CORNER_BOT_LEFT_75 ||
               LMARK[l] == LMARK_CORNER_BOT_RIGHT_25 || LMARK[l] == LMARK_CORNER_BOT_RIGHT_75 ||
               LMARK[l] == LMARK_CORNER_TOP_LEFT_25 || LMARK[l] == LMARK_CORNER_TOP_LEFT_75 ||
               LMARK[l] == LMARK_CORNER_TOP_RIGHT_25 || LMARK[l] == LMARK_CORNER_TOP_RIGHT_75) {  // no-slip on corners
        U[NMAX + l] = 0;
        U[2 * NMAX + l] = 0;
    }
}
