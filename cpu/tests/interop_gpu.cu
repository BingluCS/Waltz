#include "waltz_cpu.h"
#include "waltz.hpp"
#include "utils/Blob.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <dlfcn.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if !defined(WALTZ_SOURCE_ROOT) || !defined(WALTZ_PRODUCTION_LIBRARY)
#error "Build this test with WALTZ_CPU_BUILD_GPU_INTEROP in cpu/CMakeLists.txt"
#endif

// This executable must only be run while holding an exclusive GPU lease. It
// links the existing production shared object; it never builds GPU codec code.
namespace {
unsigned checks = 0, failures = 0, cases = 0;
bool legacy_profile = false;
const char* dump_prefix = nullptr;

void check(bool good, const char* what) {
    ++checks;
    if (!good) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
bool cuda_ok(cudaError_t rc, const char* what) {
    if (rc != cudaSuccess) std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(rc));
    check(rc == cudaSuccess, what);
    return rc == cudaSuccess;
}

struct Host {
    unsigned char* data;
    size_t bytes;
    explicit Host(size_t n) : data(static_cast<unsigned char*>(std::malloc(n))), bytes(n) {
        if (!data) { std::fprintf(stderr, "host allocation failed\n"); std::exit(2); }
    }
    ~Host() { std::free(data); }
    Host(const Host&) = delete;
    Host& operator=(const Host&) = delete;
};
struct Device {
    void* data = nullptr;
    explicit Device(size_t n) {
        if (!cuda_ok(cudaMalloc(&data, n), "device allocation")) throw "device allocation failed";
    }
    ~Device() { if (data) cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
struct Workspace {
    waltz_cpu_workspace* data = waltz_cpu_create();
    Workspace() { if (!data) { std::fprintf(stderr, "workspace allocation failed\n"); std::exit(2); } }
    ~Workspace() { waltz_cpu_destroy(data); }
};

// Hash the actual loaded file, not just a pathname supplied to the linker. Use
// argv rather than a shell so spaces/metacharacters in a path remain harmless.
bool sha256_file(const char* path, char result[65]) {
    int fd[2];
    if (pipe(fd)) return false;
    const pid_t pid = fork();
    if (pid == 0) {
        close(fd[0]);
        if (dup2(fd[1], STDOUT_FILENO) < 0) _exit(127);
        close(fd[1]);
        execlp("sha256sum", "sha256sum", "--", path, static_cast<char*>(nullptr));
        _exit(127);
    }
    close(fd[1]);
    if (pid < 0) { close(fd[0]); return false; }
    size_t kept = 0;
    char block[512];
    ssize_t got;
    while ((got = read(fd[0], block, sizeof(block))) > 0) {
        for (ssize_t i = 0; i < got && kept < 64; ++i) result[kept++] = block[i];
    }
    close(fd[0]);
    int status = 0;
    if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) || kept != 64) return false;
    result[64] = 0;
    for (unsigned i = 0; i < 64; ++i)
        if (!((result[i] >= '0' && result[i] <= '9') || (result[i] >= 'a' && result[i] <= 'f'))) return false;
    return true;
}

bool identities() {
    Dl_info info{};
    const auto symbol = &WALTZ::d_WALTZ_compress<float, 3>;
    if (!dladdr(reinterpret_cast<const void*>(symbol), &info) || !info.dli_fname) {
        std::fprintf(stderr, "cannot resolve loaded GPU library identity\n"); return false;
    }
    char actual[PATH_MAX], expected[PATH_MAX], hash[65];
    if (!realpath(info.dli_fname, actual) || !realpath(WALTZ_PRODUCTION_LIBRARY, expected)) return false;
    check(std::strcmp(actual, expected) == 0, "loaded exact production GPU shared object");
    if (std::strcmp(actual, expected) != 0) {
        std::fprintf(stderr, "loaded=%s expected=%s\n", actual, expected); return false;
    }
    if (!sha256_file(actual, hash)) return false;
    std::printf("IDENTITY loaded_library=%s sha256=%s\n", actual, hash);
    const char* sources[] = {"src/waltz.cu", "include/utils/Blob.hpp", "include/transformers/all_halo.cuh",
                              "include/transformers/halo3d.cuh", "src/small_pwe.cuh", "src/native_repairs.cuh"};
    for (const char* source : sources) {
        char path[PATH_MAX];
        std::snprintf(path, sizeof(path), "%s/%s", WALTZ_SOURCE_ROOT, source);
        if (!sha256_file(path, hash)) return false;
        std::printf("IDENTITY current_source=%s sha256=%s\n", source, hash);
    }
    std::printf("IDENTITY note=current-source hashes describe the audited files, not proven build provenance of the preexisting library\n");
    return true;
}

template <typename T>
void fill(T* values, const waltz_cpu_config& c, unsigned pattern) {
    uint32_t rng = 0x39a4b1d7;
    for (uint32_t z = 0; z < c.nz; ++z)
        for (uint32_t y = 0; y < c.ny; ++y)
            for (uint32_t x = 0; x < c.nx; ++x) {
                const size_t i = (size_t(z) * c.ny + y) * c.nx + x;
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                double value;
                if (pattern == 0) value = 0;
                else if (pattern == 1) value = 1.25;
                else if (pattern == 3) value = double(rng & 0xffffffU) / double(1U << 23) - 1;
                else if (pattern == 4) value = ((x + y + z) & 1) ? -4 : 4;
                else value = std::sin(0.17 * x) + std::cos(0.21 * y) - 0.03 * z +
                             ((x + 3 * y + 7 * z) % 19 == 0 ? 0.125 : 0);
                values[i] = static_cast<T>(value);
            }
}

template <typename T>
bool bounds(const T* input, const T* decoded, size_t n, double bound, const char* label) {
    bool good = true;
    long double maximum = 0;
    for (size_t i = 0; i < n; ++i) {
        const long double error = std::fabs(static_cast<long double>(input[i]) - decoded[i]);
        if (!std::isfinite(decoded[i]) || !(error <= static_cast<long double>(bound))) {
            if (good) std::fprintf(stderr, "  %s bound violation index=%zu input=%.17g output=%.17g error=%.21Lg bound=%.17g\n",
                                   label, i, double(input[i]), double(decoded[i]), error, bound);
            good = false;
        }
        if (error > maximum) maximum = error;
    }
    std::printf("  BOUND %s max_error=%.21Lg eb=%.17g result=%s\n", label, maximum, bound, good ? "PASS" : "FAIL");
    check(good, label);
    return good;
}

template <typename T>
bool same_bits(const T* cpu, const T* gpu, size_t n, const char* label, const Host* wire = nullptr) {
    size_t mismatches = 0, first = 0;
    size_t repaired_mismatches = 0;
    auto* repaired = static_cast<unsigned char*>(wire ? std::calloc(n, 1) : nullptr);
    if (wire && !repaired) std::exit(2);
    if (wire) {
        WALTZ::WaltzBlobHeader h{};
        std::memcpy(&h, wire->data, sizeof(h));
        const size_t at = (WALTZ::BlobSize + size_t(h.mag_comp_bytes) + 7) & ~size_t(7);
        for (uint32_t j = 0; j < h.n_pwe; ++j) {
            uint32_t index = 0;
            std::memcpy(&index, wire->data + at + size_t(j) * 4, 4);
            if (index < n) repaired[index] = 1;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(cpu + i, gpu + i, sizeof(T))) {
            if (!mismatches) first = i;
            ++mismatches;
            if (repaired && repaired[i]) ++repaired_mismatches;
        }
    }
    if (mismatches) {
        uint64_t a = 0, b = 0;
        std::memcpy(&a, cpu + first, sizeof(T)); std::memcpy(&b, gpu + first, sizeof(T));
        std::fprintf(stderr, "  %s bit mismatch count=%zu first=%zu cpu=%.17g gpu=%.17g cpu_bits=%016llx gpu_bits=%016llx\n",
                     label, mismatches, first, double(cpu[first]), double(gpu[first]),
                     static_cast<unsigned long long>(a), static_cast<unsigned long long>(b));
    }
    std::printf("  BITS %s mismatches=%zu result=%s\n", label, mismatches, mismatches ? "FAIL" : "PASS");
    if (wire && mismatches)
        std::printf("  MISMATCH_LOCATION repaired=%zu unrepaired=%zu first_repaired=%u\n",
                    repaired_mismatches, mismatches - repaired_mismatches, unsigned(repaired[first]));
    std::free(repaired);
    check(mismatches == 0, label);
    return mismatches == 0;
}

void header(const Host& archive, size_t written, const char* origin) {
    WALTZ::WaltzBlobHeader h{};
    if (written >= sizeof(h)) std::memcpy(&h, archive.data, sizeof(h));
    std::printf("  ARCHIVE origin=%s bytes=%zu synthesis=%u dyadic=%u levels=%u/%u bz=%u pwe_encoding=%u pwe_bytes=%u n_pwe=%u n_qo=%u q=%.17g\n",
                origin, written, unsigned(h.synthesis()), h.use_dyadic(), h.levels_xy, h.levels_z,
                h.reorder_bz, unsigned(h.pwe_encoding()), h.pwe_bytes, h.n_pwe, h.n_qo, h.q);
}

template <typename T>
void dump(const Host& archive, size_t bytes, uint32_t nx, uint32_t ny, uint32_t nz,
          unsigned pattern, double bound, const char* origin) {
    if (!dump_prefix) return;
    char path[PATH_MAX];
    uint64_t bound_bits = 0;
    std::memcpy(&bound_bits, &bound, sizeof(bound_bits));
    const int length = std::snprintf(path, sizeof(path), "%s_f%zu_%ux%ux%u_p%u_eb%016llx_%s.wlt2",
                                     dump_prefix, sizeof(T) * 8, nx, ny, nz, pattern,
                                     static_cast<unsigned long long>(bound_bits), origin);
    if (length < 0 || size_t(length) >= sizeof(path)) { check(false, "archive dump path length"); return; }
    FILE* file = std::fopen(path, "wb");
    check(file != nullptr, "open requested archive dump");
    if (!file) return;
    const bool good = std::fwrite(archive.data, 1, bytes, file) == bytes;
    const bool closed = std::fclose(file) == 0;
    check(good && closed, "write requested archive dump");
    std::printf("  FILE origin=%s path=%s bytes=%zu\n", origin, path, bytes);
}

template <typename T>
void run_case(Workspace& w, uint32_t nx, uint32_t ny, uint32_t nz, unsigned pattern, double bound) {
    ++cases;
    const unsigned before = failures;
    const size_t n = size_t(nx) * ny * nz, bytes = n * sizeof(T);
    waltz_cpu_config cc{nx, ny, nz, sizeof(T) == 4 ? WALTZ_CPU_F32 : WALTZ_CPU_F64, bound, WALTZ_CPU_SCALAR};
    WALTZ::Config gc;
    // Config's variadic constructor drops singleton dimensions. Set the exact
    // 3-D shape directly so tiny axes exercise the actual 3-D codec path.
    gc.ndims = 3; gc.dimx = nx; gc.dimy = ny; gc.dimz = nz; gc.num = n;
    gc.errorBoundMode = WALTZ::EB_ABS; gc.absErrorBound = bound;
    const size_t cpu_capacity = waltz_cpu_max_compressed_bytes(&cc);
    const size_t gpu_capacity = WALTZ::compressed_buffer_capacity<T>(n);
    const size_t capacity = cpu_capacity > gpu_capacity ? cpu_capacity : gpu_capacity;
    Host input(bytes), saved(bytes), cpu_decoded(bytes), gpu_decoded(bytes), alternate(bytes), archive(capacity), cpu_archive(capacity);
    Device device_input(bytes), device_output(bytes), device_archive(capacity);
    fill(reinterpret_cast<T*>(input.data), cc, pattern);
    std::memcpy(saved.data, input.data, bytes);
    std::printf("CASE f%zu %ux%ux%u pattern=%u eb=%.17g\n", 8 * sizeof(T), nx, ny, nz, pattern, bound);

    // CPU -> GPU. Each decoder sees this very same byte stream. Comparing
    // independently encoded streams would conflate atomic residual ordering
    // with a decoding/arithmetic interoperability defect.
    size_t written = 0;
    int rc = waltz_cpu_compress(w.data, &cc, input.data, archive.data, archive.bytes, &written);
    if (rc) std::fprintf(stderr, "CPU encode: %s\n", waltz_cpu_last_error(w.data));
    check(rc == 0 && written <= capacity, "CPU archive encode");
    if (rc || written > capacity) return;
    header(archive, written, "CPU");
    dump<T>(archive, written, nx, ny, nz, pattern, bound, "CPU");
    const size_t cpu_written = written;
    std::memcpy(cpu_archive.data, archive.data, cpu_written);
    check(std::memcmp(input.data, saved.data, bytes) == 0, "CPU encode preserves input");
    rc = waltz_cpu_decompress(w.data, &cc, archive.data, written, cpu_decoded.data, bytes);
    if (rc) std::fprintf(stderr, "CPU decode CPU stream: %s\n", waltz_cpu_last_error(w.data));
    check(rc == 0, "CPU decode CPU archive");
    if (rc) return;
    if (!cuda_ok(cudaMemcpy(device_archive.data, archive.data, written, cudaMemcpyHostToDevice), "upload CPU archive")) return;
    const size_t gpu_bytes = WALTZ::d_decompress<T>(gc, static_cast<char*>(device_archive.data),
                                                  static_cast<T*>(device_output.data), nullptr, nullptr);
    check(gpu_bytes == bytes, "GPU decode CPU archive output length");
    if (!cuda_ok(cudaMemcpy(gpu_decoded.data, device_output.data, bytes, cudaMemcpyDeviceToHost), "download CPU->GPU reconstruction")) return;
    same_bits(reinterpret_cast<T*>(cpu_decoded.data), reinterpret_cast<T*>(gpu_decoded.data), n, "CPU archive CPU/GPU decode", &archive);
    bounds(reinterpret_cast<T*>(input.data), reinterpret_cast<T*>(cpu_decoded.data), n, bound, "CPU archive CPU strict bound");
    bounds(reinterpret_cast<T*>(input.data), reinterpret_cast<T*>(gpu_decoded.data), n, bound, "CPU archive GPU strict bound");

    if (waltz_cpu_has_avx2()) {
        cc.backend = WALTZ_CPU_AVX2;
        size_t avx_written = 0;
        rc = waltz_cpu_compress(w.data, &cc, input.data, archive.data, capacity, &avx_written);
        check(rc == 0, "AVX2 archive encode");
        check(!rc && avx_written == cpu_written && !std::memcmp(archive.data, cpu_archive.data, cpu_written),
              "scalar/AVX2 complete encoded archive bytes");
        if (!rc) dump<T>(archive, avx_written, nx, ny, nz, pattern, bound, "AVX2");
        cc.backend = WALTZ_CPU_SCALAR;
    }

    // The ordinary encoder must emit CPU-identical bytes with default settings.
    if (!cuda_ok(cudaMemcpy(device_input.data, input.data, bytes, cudaMemcpyHostToDevice), "upload original data")) return;
    const auto start = std::chrono::steady_clock::now();
    written = WALTZ::d_compress<T>(gc, static_cast<T*>(device_input.data),
                                  static_cast<char*>(device_archive.data), nullptr, nullptr);
    const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::printf("  WALL GPU_d_compress_ms=%.6f\n", wall_ms);
    check(written >= WALTZ::BlobSize && written <= capacity, "GPU archive length");
    if (written > capacity || written < WALTZ::BlobSize) return;
    if (!cuda_ok(cudaMemcpy(archive.data, device_archive.data, written, cudaMemcpyDeviceToHost), "download GPU archive")) return;
    header(archive, written, "GPU");
    dump<T>(archive, written, nx, ny, nz, pattern, bound, "GPU");
    if (!legacy_profile) {
        const bool bytes_equal = written == cpu_written && !std::memcmp(archive.data, cpu_archive.data, written);
        check(bytes_equal, "CPU/scalar/AVX2/GPU complete encoded archive bytes");
        if (!bytes_equal) {
            const size_t common = written < cpu_written ? written : cpu_written;
            size_t first = 0;
            while (first < common && archive.data[first] == cpu_archive.data[first]) ++first;
            std::fprintf(stderr, "  archive mismatch CPU_bytes=%zu GPU_bytes=%zu first_difference=%zu\n", cpu_written, written, first);
            header(cpu_archive, cpu_written, "CPU-reference");
        }
        // Poison unused storage before encoding again so nondeterministic
        // ordering or uncleared padding cannot accidentally compare equal.
        if (!cuda_ok(cudaMemset(device_archive.data, 0xa5, capacity), "poison archive")) return;
        const size_t repeat = WALTZ::d_compress<T>(gc, static_cast<T*>(device_input.data),
                                                  static_cast<char*>(device_archive.data), nullptr, nullptr);
        if (!cuda_ok(cudaMemcpy(archive.data, device_archive.data, written, cudaMemcpyDeviceToHost), "download repeated encoding")) return;
        check(repeat == cpu_written && written == cpu_written && !std::memcmp(archive.data, cpu_archive.data, written),
              "repeated ordinary encoding matches CPU archive");
    }
    const size_t gpu_bytes2 = WALTZ::d_decompress<T>(gc, static_cast<char*>(device_archive.data),
                                                   static_cast<T*>(device_output.data), nullptr, nullptr);
    check(gpu_bytes2 == bytes, "GPU decode GPU archive output length");
    if (!cuda_ok(cudaMemcpy(gpu_decoded.data, device_output.data, bytes, cudaMemcpyDeviceToHost), "download GPU->GPU reconstruction")) return;
    rc = waltz_cpu_decompress(w.data, &cc, archive.data, written, cpu_decoded.data, bytes);
    if (rc) std::fprintf(stderr, "CPU decode GPU stream: %s\n", waltz_cpu_last_error(w.data));
    check(rc == 0, "CPU decode GPU archive");
    if (!rc) {
        same_bits(reinterpret_cast<T*>(cpu_decoded.data), reinterpret_cast<T*>(gpu_decoded.data), n, "GPU archive CPU/GPU decode", &archive);
        bounds(reinterpret_cast<T*>(input.data), reinterpret_cast<T*>(cpu_decoded.data), n, bound, "GPU archive CPU strict bound");
        if (waltz_cpu_has_avx2()) {
            cc.backend = WALTZ_CPU_AVX2;
            rc = waltz_cpu_decompress(w.data, &cc, archive.data, written, alternate.data, bytes);
            check(rc == 0, "AVX2 decode GPU archive");
            if (!rc) same_bits(reinterpret_cast<T*>(cpu_decoded.data), reinterpret_cast<T*>(alternate.data), n,
                               "GPU archive scalar/AVX2 decode");
        }
    }
    bounds(reinterpret_cast<T*>(input.data), reinterpret_cast<T*>(gpu_decoded.data), n, bound, "GPU archive GPU strict bound");
    std::printf("RESULT f%zu %ux%ux%u pattern=%u %s\n", 8 * sizeof(T), nx, ny, nz, pattern,
                failures == before ? "PASS" : "FAIL");
}

template <typename T>
void suite(Workspace& w, bool full) {
    run_case<T>(w, 2, 3, 5, 2, 0.001);
    run_case<T>(w, 16, 16, 16, 2, 0.001);
    run_case<T>(w, 17, 19, 23, 2, 0.001);
    run_case<T>(w, 48, 32, 9, 2, 0.001);
    run_case<T>(w, 9, 11, 65, 3, 0.001);
    run_case<T>(w, 16, 16, 16, 0, 0.001);
    run_case<T>(w, 17, 19, 23, 1, 0.001);
    if (full) {
        run_case<T>(w, 1, 1, 1, 2, 0.001);
        run_case<T>(w, 1, 17, 33, 2, 0.001);
        run_case<T>(w, 33, 31, 17, 4, 0.001);
        run_case<T>(w, 64, 64, 64, 2, 0.001);
        run_case<T>(w, 128, 96, 65, 3, 0.0001);
        // Eligible for optional compact fused Z; the default must match CPU.
        if constexpr (std::is_same_v<T,float>)
            run_case<T>(w, 72, 70, 25, 2, 0.001);
        // Exercise nonempty quantization-outlier index/value radix sorting;
        // the ordinary error bounds above usually keep all magnitudes u16.
        run_case<T>(w, 17, 19, 23, 2, 0.00001);
    }
}
} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    int device = 0;
    bool full = false, identity_only = false, custom = false, f64 = false;
    uint32_t nx = 16, ny = 16, nz = 16;
    unsigned pattern = 2;
    double bound = 0.001;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--device") && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--full")) full = true;
        else if (!std::strcmp(argv[i], "--legacy")) legacy_profile = true;
        else if (!std::strcmp(argv[i], "--dump-prefix") && i + 1 < argc) dump_prefix = argv[++i];
        else if (!std::strcmp(argv[i], "--identity-only")) identity_only = true;
        else if (!std::strcmp(argv[i], "--case") && i + 6 < argc) {
            custom = true; f64 = !std::strcmp(argv[++i], "f64");
            nx = uint32_t(std::strtoul(argv[++i], nullptr, 10));
            ny = uint32_t(std::strtoul(argv[++i], nullptr, 10));
            nz = uint32_t(std::strtoul(argv[++i], nullptr, 10));
            bound = std::strtod(argv[++i], nullptr); pattern = unsigned(std::strtoul(argv[++i], nullptr, 10));
        } else {
            std::fprintf(stderr, "usage: %s [--device N] [--full] [--legacy] [--identity-only] [--dump-prefix PATH] [--case f32|f64 NX NY NZ EB PATTERN]\n", argv[0]);
            return 2;
        }
    }
    if (!identities()) { std::fprintf(stderr, "GPU/source identity check failed\n"); return 2; }
    if (identity_only) return failures ? 1 : 0; // Does not call the CUDA API.
    if (!legacy_profile) {
        unsetenv("WALTZ_HALO3D");
        unsetenv("WALTZ_SMALL_PWE_SCALE");
    }
    if (!nx || !ny || !nz || !(bound > 0) || !std::isfinite(bound)) return 2;
    if (!cuda_ok(cudaSetDevice(device), "select exclusively leased GPU")) return 2;
    cudaDeviceProp prop{};
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "read leased GPU identity")) return 2;
    std::printf("DEVICE id=%d name=%s cc=%d.%d\n", device, prop.name, prop.major, prop.minor);
    const char* modes[] = {"WALTZ_HALO3D", "WALTZ_SMALL_PWE_SCALE"};
    for (const char* name : modes) std::printf("ENV %s=%s\n", name, std::getenv(name) ? std::getenv(name) : "<default>");
    Workspace w;
    try {
        if (custom) {
            if (f64) run_case<double>(w, nx, ny, nz, pattern, bound);
            else run_case<float>(w, nx, ny, nz, pattern, bound);
        } else { suite<float>(w, full); suite<double>(w, full); }
    } catch (const std::exception& error) {
        ++failures; std::fprintf(stderr, "GPU codec exception: %s\n", error.what());
    } catch (const char* error) {
        ++failures; std::fprintf(stderr, "%s\n", error);
    }
    std::printf("SUMMARY cases=%u checks=%u failures=%u\n", cases, checks, failures);
    return failures ? 1 : 0;
}
