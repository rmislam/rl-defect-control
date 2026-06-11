#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
echo "Compiling..."
nvcc -o ../src/cpp/lbm_cuda ../src/cpp/main.cu ../src/cpp/cuda_params.cu -lm -lpthread --use_fast_math -Xcompiler -Ofast,-march=native -arch=sm_120
echo "Running simulation..."
cd ../src/cpp && ./lbm_cuda
cd "$parent_path"
#echo "Profiling..."
#cd ../src/cpp && ncu -o profile -f --launch-skip 100 --launch-count 32 --section SchedulerStats --section WarpStateStats ./lbm_cuda
echo "Done"
