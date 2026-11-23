#include "waltz_cpu.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

// Standalone full-C-API benchmark. All caller allocations are outside the
// timed calls; one workspace is reused across warmup, repetitions and backends.
// No STL containers are used. Throughput is uncompressed MiB per wall second.
namespace {

using Clock = std::chrono::steady_clock;
constexpr unsigned repeats = 2;
constexpr double absolute_error = 0.001;

struct Buffer {
    unsigned char* base;
    unsigned char* data;
    std::size_t bytes;
    explicit Buffer(std::size_t n)
        : base(static_cast<unsigned char*>(std::malloc(n + 128))),
          data(base ? base + 64 : nullptr), bytes(n) {
        if (!base) throw std::bad_alloc();
        std::memset(base, 0xa5, n + 128);
    }
    ~Buffer() { std::free(base); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    bool guards() const {
        for (unsigned i = 0; i < 64; ++i)
            if (base[i] != 0xa5 || data[bytes + i] != 0xa5) return false;
        return true;
    }
};

struct Workspace {
    waltz_cpu_workspace* value;
    Workspace() : value(waltz_cpu_create()) {
        if (!value) throw std::bad_alloc();
    }
    ~Workspace() { waltz_cpu_destroy(value); }
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void api_result(int result, const Workspace& workspace, const char* operation) {
    if (result) {
        std::fprintf(stderr, "%s: %s\n", operation, waltz_cpu_last_error(workspace.value));
        throw std::runtime_error("codec API failed");
    }
}

template<class T>
void fill(T* data, const waltz_cpu_config& config) {
    std::uint32_t rng = 0x7193ae21U;
    for (std::uint32_t z = 0; z < config.nz; ++z)
        for (std::uint32_t y = 0; y < config.ny; ++y)
            for (std::uint32_t x = 0; x < config.nx; ++x) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                const double noise = (double(rng & 65535U) / 65536.0 - 0.5) * 0.002;
                const double value = std::sin(0.071 * x) + 0.7 * std::cos(0.053 * y)
                                   + 0.4 * std::sin(0.117 * z) + 0.002 * x - 0.001 * y + noise;
                data[(std::size_t(z) * config.ny + y) * config.nx + x] = static_cast<T>(value);
            }
}

struct Quality {
    double psnr;
    double maximum;
};

template<class T>
Quality quality(const T* input, const T* output, std::size_t n) {
    long double minimum = input[0], maximum = input[0], squared = 0, max_error = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const long double sample = input[i];
        const long double difference = std::fabs(sample - static_cast<long double>(output[i]));
        require(std::isfinite(output[i]), "nonfinite reconstruction");
        require(difference <= static_cast<long double>(absolute_error), "strict pointwise bound exceeded");
        if (sample < minimum) minimum = sample;
        if (sample > maximum) maximum = sample;
        if (difference > max_error) max_error = difference;
        squared += difference * difference;
    }
    const long double range = maximum - minimum;
    const long double rmse = std::sqrt(squared / n);
    const double psnr = rmse == 0 ? std::numeric_limits<double>::infinity()
                                 : double(20 * std::log10(range / rmse));
    return {psnr, double(max_error)};
}

template<class T>
void benchmark(std::uint32_t nx, std::uint32_t ny, std::uint32_t nz, bool dyadic) {
    waltz_cpu_config config{nx, ny, nz, sizeof(T) == 4 ? WALTZ_CPU_F32 : WALTZ_CPU_F64,
                            absolute_error, WALTZ_CPU_SCALAR};
    const std::size_t count = std::size_t(nx) * ny * nz, bytes = count * sizeof(T);
    const std::size_t capacity = waltz_cpu_max_compressed_bytes(&config);
    require(capacity >= 40, "invalid maximum archive capacity");
    Buffer input(bytes), saved_input(bytes), output(bytes), scalar_output(bytes);
    Buffer archive(capacity), scalar_archive(capacity);
    fill(reinterpret_cast<T*>(input.data), config);
    std::memcpy(saved_input.data, input.data, bytes);
    Workspace workspace;
    std::size_t scalar_bytes = 0;
    const waltz_cpu_backend backends[] = {WALTZ_CPU_SCALAR, WALTZ_CPU_AVX2};
    for (waltz_cpu_backend backend : backends) {
        if (backend == WALTZ_CPU_AVX2 && !waltz_cpu_has_avx2()) continue;
        config.backend = backend;
        double compress_seconds = 0, decompress_seconds = 0;
        std::size_t archive_bytes = 0;
        Quality result{};
        for (unsigned run = 0; run <= repeats; ++run) {
            const auto begin_compress = Clock::now();
            const int compressed = waltz_cpu_compress(workspace.value, &config, input.data,
                                                      archive.data, capacity, &archive_bytes);
            const auto end_compress = Clock::now();
            api_result(compressed, workspace, "compress");
            require(archive_bytes >= 40 && archive_bytes <= capacity, "invalid encoded size");
            require(archive.data[8] == unsigned(dyadic), "unexpected transform schedule");
            const auto begin_decompress = Clock::now();
            const int decompressed = waltz_cpu_decompress(workspace.value, &config, archive.data,
                                                          archive_bytes, output.data, bytes);
            const auto end_decompress = Clock::now();
            api_result(decompressed, workspace, "decompress");
            if (run) {
                compress_seconds += std::chrono::duration<double>(end_compress - begin_compress).count();
                decompress_seconds += std::chrono::duration<double>(end_decompress - begin_decompress).count();
            }
            result = quality(reinterpret_cast<const T*>(input.data), reinterpret_cast<const T*>(output.data), count);
            require(std::memcmp(input.data, saved_input.data, bytes) == 0, "caller input modified");
            require(input.guards() && saved_input.guards() && output.guards() && archive.guards(), "caller buffer guard corrupted");
            if (backend == WALTZ_CPU_SCALAR && !run) {
                scalar_bytes = archive_bytes;
                std::memcpy(scalar_archive.data, archive.data, archive_bytes);
                std::memcpy(scalar_output.data, output.data, bytes);
            } else {
                require(archive_bytes == scalar_bytes && !std::memcmp(archive.data, scalar_archive.data, scalar_bytes),
                        "scalar/AVX2 or repeated archive bytes differ");
                require(!std::memcmp(output.data, scalar_output.data, bytes), "scalar/AVX2 or repeated decode bits differ");
            }
        }
        compress_seconds /= repeats;
        decompress_seconds /= repeats;
        std::printf("f%zu %ux%ux%u %-6s %-6s bytes=%zu CR=%.3f PSNR_dB=%.3f max_abs=%.9g "
                    "compress_ms=%.3f decompress_ms=%.3f compress_MiB_s=%.3f decompress_MiB_s=%.3f "
                    "strict=PASS archive_memcmp=PASS decode_memcmp=PASS\n",
                    sizeof(T) * 8, nx, ny, nz, dyadic ? "dyadic" : "plane",
                    backend == WALTZ_CPU_SCALAR ? "scalar" : "avx2", archive_bytes,
                    double(bytes) / archive_bytes, result.psnr, result.maximum,
                    compress_seconds * 1000, decompress_seconds * 1000,
                    double(bytes) / (1048576 * compress_seconds), double(bytes) / (1048576 * decompress_seconds));
    }
}

} // namespace

int main(int argc,char** argv) {
    try {
        std::printf("Waltz CPU full C API synthetic benchmark: absolute_error=%.3g, warmup=1, repeats=%u, AVX2/FMA=%s\n",
                    absolute_error, repeats, waltz_cpu_has_avx2() ? "yes" : "no (AVX2 skipped)");
        if(argc==2&&!std::strcmp(argv[1],"--large")) {
            benchmark<float>(128,128,128,true);
            benchmark<double>(128,128,128,true);
            benchmark<float>(512,512,8,false);
            benchmark<double>(512,512,8,false);
            return 0;
        }
        if(argc==2&&!std::strcmp(argv[1],"--volume")) {
            benchmark<float>(256,256,256,true);
            benchmark<double>(256,256,256,true);
            return 0;
        }
        if(argc!=1) {std::fprintf(stderr,"Usage: %s [--large|--volume]\n",argv[0]);return 2;}
        benchmark<float>(64, 64, 64, true);
        benchmark<double>(64, 64, 64, true);
        benchmark<float>(128, 128, 8, false);
        benchmark<double>(128, 128, 8, false);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "benchmark FAILED: %s\n", error.what());
        return 1;
    }
}
