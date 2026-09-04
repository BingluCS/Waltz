// Standalone tuner validation harness: load a field ONCE, run the fast autotuner for
// every requested REL bound in-process (no compression pipeline, no per-bound reload).
// Usage: tune_demo <file> <f|d> <X> <Y> <Z> <rel1> [rel2 ...]
#include "waltz.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

template <typename T>
static int run(const char* path, uint32_t X, uint32_t Y, uint32_t Z, int nb, char** bounds) {
    const size_t n = static_cast<size_t>(X) * Y * Z;
    std::vector<T> h(n);
    FILE* fp = std::fopen(path, "rb");
    if (!fp || std::fread(h.data(), sizeof(T), n, fp) != n) {
        std::fprintf(stderr, "read fail %s\n", path);
        return 1;
    }
    std::fclose(fp);
    T* d = nullptr;
    cudaMalloc(&d, n * sizeof(T));
    cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    double dmin = 0.0, dmax = 0.0;
    WALTZ::d_extrema<T>(d, n, dmin, dmax, nullptr);
    const double range = dmax - dmin;
    for (int i = 0; i < nb; ++i) {
        const double rel = std::atof(bounds[i]);
        double ms = 0.0;
        const int mode = WALTZ::d_autotune_fast<T>(d, X, Y, Z, rel * range, rel, nullptr, &ms);
        static const char* names[3] = {"DYADIC", "XY+Z", "X+Y+Z"};
        std::printf("RESULT\t%s\t%s\t%s\t%.3f\n", path, bounds[i], names[mode], ms);
    }
    cudaFree(d);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s <file> <f|d> X Y Z rel...\n", argv[0]);
        return 1;
    }
    const uint32_t X = std::atoi(argv[3]), Y = std::atoi(argv[4]), Z = std::atoi(argv[5]);
    if (argv[2][0] == 'd')
        return run<double>(argv[1], X, Y, Z, argc - 6, argv + 6);
    return run<float>(argv[1], X, Y, Z, argc - 6, argv + 6);
}
