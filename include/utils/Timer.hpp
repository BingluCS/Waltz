#pragma once

#include <cuda_runtime.h>

struct GPUTimer {
    cudaEvent_t beg, end;
    GPUTimer() {
        cudaEventCreate(&beg);
        cudaEventCreate(&end);
    }
    ~GPUTimer() {
        cudaEventDestroy(beg);
        cudaEventDestroy(end);
    }
    void start(void* stream) {
        cudaEventRecord(beg, (cudaStream_t)stream);
    }
    void record_stop(void* stream) {
        cudaEventRecord(end, (cudaStream_t)stream);
    }
    double elapsed_ready() {
        float ms;
        cudaEventElapsedTime(&ms, beg, end);
        return ms;
    }

};
