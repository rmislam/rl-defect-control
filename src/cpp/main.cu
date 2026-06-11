// Modified and extended by Russ Islam
// Original tutorial code created by Žiga Kos and Miha Ravnik: https://zenodo.org/records/4737814

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Declare host arrays
float *h_U, *h_F, *h_FEQ, *h_FNEW, *h_P, *h_Q, *h_QNEW, *h_H, *h_SIGMA;
float *h_ACTIVITY, *h_DEFECT_MAP;
char *h_LMARK;
float *h_U_INIT, *h_F_INIT, *h_Q_INIT; // for sim reset
float *h_ACTIVITY_INIT; // for sim reset

// Declare device arrays
float *U, *F, *FEQ, *FNEW, *P, *Q, *QNEW, *H, *SIGMA;
float *ACTIVITY;
char *LMARK;

#include "headers.h"
#include "cuda_params.h"
#include "parameters.c"
#include "utility_functions.c"
#include "geometry.c"
#include "lattice_Boltzmann.cu"
#include "finite_difference.cu"
#include "defect.c"


int main(int argc, char** args) {
    clock_t start_time, end_time;
    float time_elapsed;
    start_time = clock();

    // Declare variables for reinforcement learning
    bool is_done = false;
    float dist_from_goal = 1e+03;
    float prev_dist_from_goal = 1e+03;
    float reward = 0;
    //float pattern_fraction = 0;

    initCUDAConstants();

    // Allocate host memory
    h_U = (float *)malloc(3 * NMAX * sizeof(float));  //velocity field; 0th component is density
    h_F = (float *)malloc(LATTICE_VELOCITY_NUMBER * NMAX * sizeof(float));  //Lattice Boltzmann distribution functions
    h_FEQ = (float *)malloc(LATTICE_VELOCITY_NUMBER * NMAX * sizeof(float));  //Lattice Boltzmann equilibrium distribution functions
    h_FNEW = (float *)malloc(LATTICE_VELOCITY_NUMBER * NMAX * sizeof(float));  //matrix field to temporarily store currently calculated F
    h_P = (float *)malloc(LATTICE_VELOCITY_NUMBER * NMAX * sizeof(float));  // Lattice Boltzmann forcing terms
    h_Q = (float *)malloc(2 * NMAX * sizeof(float));  //Q-tensor
    h_QNEW = (float *)malloc(2 * NMAX * sizeof(float));  //Q-tensor to temporarily store values
    h_H = (float *)malloc(2 * NMAX * sizeof(float));  //Molecular field (from Q-tensor relaxation)
    h_SIGMA = (float *)malloc(4 * NMAX * sizeof(float));  //Stress tensor
    h_LMARK = (char *)malloc(NMAX * sizeof(char));        //Logical markers to determine a mesh point function (bulk or boundary condition)
    h_ACTIVITY = (float *)malloc(NMAX * sizeof(float));  // activity vs. no activity
    h_DEFECT_MAP = (float *)malloc(NMAX * sizeof(float));  // 1.0 for +1/2 defects, -1.0 for -1/2 defects, 0 elsewhere
    h_U_INIT = (float *)malloc(3 * NMAX * sizeof(float));  //velocity field; 0th component is density (for sim reset)
    h_F_INIT = (float *)malloc(LATTICE_VELOCITY_NUMBER * NMAX * sizeof(float));  // Lattice Boltzmann distribution functions (for sim reset)
    h_Q_INIT = (float *)malloc(2 * NMAX * sizeof(float));  //Q-tensor (for sim reset)
    h_ACTIVITY_INIT = (float *)malloc(NMAX * sizeof(float));  // activity vs. no activity (for sim reset)

    // Allocate device memory
    cudaMalloc(&U, 3 * NMAX * sizeof(float));  //velocity field; 0th component is density
    cudaMalloc(&F, LATTICE_VELOCITY_NUMBER * NMAX * sizeof(float));  //Lattice Boltzmann distribution functions
    cudaMalloc(&FEQ, LATTICE_VELOCITY_NUMBER * NMAX * sizeof(float));  //Lattice Boltzmann equilibrium distribution functions
    cudaMalloc(&FNEW, LATTICE_VELOCITY_NUMBER * NMAX * sizeof(float));  //matrix field to temporarily store currently calculated F
    cudaMalloc(&P, LATTICE_VELOCITY_NUMBER * NMAX * sizeof(float));  // Lattice Boltzmann forcing terms
    cudaMalloc(&Q, 2 * NMAX * sizeof(float));  //Q-tensor
    cudaMalloc(&QNEW, 2 * NMAX * sizeof(float));  //Q-tensor to temporarily store values
    cudaMalloc(&H, 2 * NMAX * sizeof(float));  //Molecular field (from Q-tensor relaxation)
    cudaMalloc(&SIGMA, 4 * NMAX * sizeof(float));  //Stress tensor
    cudaMalloc(&LMARK, NMAX * sizeof(char));        //Logical markers to determine a mesh point function (bulk or boundary condition)
    cudaMalloc(&ACTIVITY, NMAX * sizeof(float));  // activity vs. no activity

    // Set random seed
    unsigned seed = 12345;
    srand(seed);

    createGeometry();

    // Copy host arrays to device
    cudaMemcpy(LMARK, h_LMARK, NMAX * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(ACTIVITY, h_ACTIVITY, NMAX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(U, h_U, 3 * NMAX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Q, h_Q, 2 * NMAX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(QNEW, h_QNEW, 2 * NMAX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(FEQ, h_FEQ, LATTICE_VELOCITY_NUMBER * NMAX * sizeof(float), cudaMemcpyHostToDevice);

    // Compute distribution functions from velocity initialization
    computeFeq<<<GRID_DIM, BLOCK_DIM>>>(U, FEQ);
    cudaMemcpy(h_FEQ, FEQ, LATTICE_VELOCITY_NUMBER * NMAX * sizeof(float), cudaMemcpyDeviceToHost);

    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        #pragma unroll
        for(int m = 0; m < LATTICE_VELOCITY_NUMBER; m++) {
            h_F[m * NMAX + l] = h_FEQ[m * NMAX + l];
            h_FNEW[m * NMAX + l] = h_F[m * NMAX + l];
        }
    }

    //Q relaxation
    printf("Starting Q relaxation without defects...\n");
    #pragma unroll
    for (int t = 0; t < TIME_PRE_EVOL; t++) {
        if (t % TIME_PRINT == 0) {
            printf("timestep: %d / %d\n", t, TIME_PRE_EVOL);
            fflush(stdout);
        }

        #pragma unroll
        for (int q_step = 0; q_step < N_EVOL_Q; q_step++) {
            compute_FD_step(U, Q, QNEW, H, LMARK);
        }
    }
    printf("Finished Q relaxation without defects\n");

    cudaMemcpy(h_Q, Q, 2 * NMAX * sizeof(float), cudaMemcpyDeviceToHost);
    addDefect();
    cudaMemcpy(Q, h_Q, 2 * NMAX * sizeof(float), cudaMemcpyHostToDevice);

    //Q relaxation
    printf("Starting Q relaxation with defects...\n");
    #pragma unroll
    for (int t = 0; t < TIME_PRE_EVOL; t++) {
        if (t % TIME_PRINT == 0) {
            printf("timestep: %d / %d\n", t, TIME_PRE_EVOL);
            fflush(stdout);
        }

        #pragma unroll
        for (int q_step = 0; q_step < N_EVOL_Q; q_step++) {
            compute_FD_step(U, Q, QNEW, H, LMARK);
        }
    }
    printf("Finished Q relaxation with defects\n");

    cudaMemcpy(h_U, U, 3 * NMAX * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Q, Q, 2 * NMAX * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_F, F, LATTICE_VELOCITY_NUMBER * NMAX * sizeof(float), cudaMemcpyDeviceToHost);
    prev_dist_from_goal = findDefects();

    // Save U, F, Q, ACTIVITY
    // No need to save P, WKONST, E, LMARK, SIGMA, H, FNEW, FEQ, QNEW
    saveInitializedState();

    //Main part - time evolution
    printf("Starting main evolution...\n");
    clock_t start_evol_time = clock();

    #pragma unroll
    for (int t = 0; t <= TIME_STEPS; t++) {
        if (t % TIME_PRINT == 0) {
            printf("timestep: %d / %d\n", t, TIME_STEPS);
            fflush(stdout);
        }

        if (t % TIME_WRITE == 0 || t == TIME_STEPS) {
            cudaMemcpy(h_U, U, 3 * NMAX * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Q, Q, 2 * NMAX * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_SIGMA, SIGMA, 4 * NMAX * sizeof(float), cudaMemcpyDeviceToHost); // NOTE: This is simply for writing out the stress

            if (WRITE_VIZ_DATA == "TRUE") {
                writeVelocity(t);
                writeOrientation(t);
                writeFreeEnergy(t);
                writeStress(t);
                writeGoal(t);
                writeActivity(t);
            }

            dist_from_goal = findDefects();
            reward = (prev_dist_from_goal - dist_from_goal) / 10.;

            if (dist_from_goal < 50) {
                reward += 0.02 * (50 - dist_from_goal);
            }

            // NOTE: For pattern regularization. Disable for now
            //pattern_fraction = computeActivityPatternFraction();
            //printf("pattern_fraction: %.3f\n", pattern_fraction);
            //reward -= pattern_fraction;  // for action regularization

            writeReward(reward);
            writeObservations();
            prev_dist_from_goal = dist_from_goal;
            printf("\nCurrent reward: %.3f\n\n", reward);
        }

        if (t == TIME_STEPS) {
            is_done = true;
        }

        if (t % TIME_SYNC == 0 || t == TIME_STEPS) {
            sendPipeSignal(is_done);
            bool reset = recvPipeSignal();  // reset == 1 means reset sim, == 0 means continue
            printf("reset: %d\n", reset);

            if (reset) {
                printf("Received reset signal\n");
                resetState();
                cudaMemcpy(ACTIVITY, h_ACTIVITY, NMAX * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(U, h_U, 3 * NMAX * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(Q, h_Q, 2 * NMAX * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(F, h_F, LATTICE_VELOCITY_NUMBER * NMAX * sizeof(float), cudaMemcpyHostToDevice);
                prev_dist_from_goal = findDefects();
                t = -1;
                is_done = false;
                continue;
            } else {
                if (t == TIME_STEPS) {
                    printf("Ending simulation\n");
                    break;
                }
            }

            readActivity();
            cudaMemcpy(ACTIVITY, h_ACTIVITY, NMAX * sizeof(float), cudaMemcpyHostToDevice);
        }

        #pragma unroll
        for (int q_step = 0; q_step < N_EVOL_Q; q_step++) {
            compute_FD_step(U, Q, QNEW, H, LMARK);
        }

        compute_LB_step(U, FNEW, F, FEQ, P, Q, H, SIGMA, ACTIVITY, LMARK);
    }

    // Deallocate device memory
    cudaFree(U);
    cudaFree(F);
    cudaFree(FEQ);
    cudaFree(FNEW);
    cudaFree(P);
    cudaFree(Q);
    cudaFree(QNEW);
    cudaFree(H);
    cudaFree(SIGMA);
    cudaFree(LMARK);
    cudaFree(ACTIVITY);

    // Deallocate host memory
    free(h_U);
    free(h_F);
    free(h_FEQ);
    free(h_FNEW);
    free(h_P);
    free(h_Q);
    free(h_QNEW);
    free(h_H);
    free(h_SIGMA);
    free(h_LMARK);
    free(h_ACTIVITY);
    free(h_DEFECT_MAP);
    free(h_U_INIT);
    free(h_F_INIT);
    free(h_Q_INIT);
    free(h_ACTIVITY_INIT);

    end_time = clock();
    time_elapsed = ((float) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Simulation took %f seconds to complete\n", time_elapsed);

    return 0;
}
