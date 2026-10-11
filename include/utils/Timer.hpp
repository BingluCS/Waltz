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

// Reuse event handles across completed API calls on the same thread/device.
// Event records and interval boundaries remain exactly where the caller puts them.
template <unsigned Count>
class GPUTimerCache {
    GPUTimer* timers = nullptr;
    int device = -1;

    void release() {
        if (timers == nullptr)
            return;
        int previous = 0;
        if (cudaGetDevice(&previous) == cudaSuccess) {
            cudaSetDevice(device);
            delete[] timers;
            cudaSetDevice(previous);
        }
        timers = nullptr;
    }

  public:
    GPUTimerCache() = default;
    GPUTimerCache(const GPUTimerCache&) = delete;
    GPUTimerCache& operator=(const GPUTimerCache&) = delete;
    ~GPUTimerCache() { release(); }
    GPUTimer* get() {
        int current = 0;
        cudaGetDevice(&current);
        if (timers == nullptr || device != current) {
            release();
            device = current;
            timers = new GPUTimer[Count];
        }
        return timers;
    }
};
