//lattice_Boltzmann.cu
void initCUDAConstants();

void compute_LB_step(float* __restrict__ U, float* __restrict__ FNEW, float* __restrict__ F, float* __restrict__ FEQ, float* __restrict__ P, float* __restrict__ Q, float* __restrict__ H, float* __restrict__ SIGMA, float* __restrict__ ACTIVITY, char* __restrict__ LMARK);

__global__
void computeFeq(float* __restrict__ U, float* __restrict__ FEQ);

__global__
void computeSigma(float* __restrict__ SIGMA, float* __restrict__ Q, float* __restrict__ H, float* __restrict__ ACTIVITY, char* __restrict__ LMARK);

__global__
void computeP(float* __restrict__ U, float* __restrict__ P, float* __restrict__ SIGMA, char* __restrict__ LMARK);

__global__
void computeFNEW(float* __restrict__ FNEW, float* __restrict__ F, float* __restrict__ FEQ, float* __restrict__ P);

__global__
void enforceBCAndCalcFNEW2F2U(float* __restrict__ FNEW, float* __restrict__ F, float* __restrict__ U, float* __restrict__ SIGMA, char* __restrict__ LMARK);

//finite_difference.c
void compute_FD_step(float* __restrict__ U, float* __restrict__ Q, float* __restrict__ QNEW, float* __restrict__ H, char* __restrict__ LMARK);

__global__
void computeQNEW(float* __restrict__ U, float* __restrict__ Q, float* __restrict__ QNEW, float* __restrict__ H, char* __restrict__ LMARK);

__global__
void calcQNEW2Q(float* __restrict__ Q, float* __restrict__ QNEW, char* __restrict__ LMARK);
