#include "waltz_cpu.h"

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {

void usage(FILE* stream, const char* program) {
    std::fprintf(stream,
        "Usage: %s c|d f32|f64 nx ny nz bound input output [scalar|avx2|auto]\n"
        "  c: compress raw samples to a canonical little-endian Waltz archive\n"
        "  d: decompress a Waltz archive to raw samples\n"
        "  nx, ny, nz: positive dimensions, with X varying fastest\n"
        "  bound: positive finite absolute error bound used for compression\n"
        "         (decode uses the quantization scale embedded in the archive)\n"
        "  backend: defaults to auto; explicit avx2 requires AVX2 and FMA\n"
        "Raw f32/f64 files use native IEEE-754 samples and the local byte order.\n"
        "Raw input size must match the shape exactly. Existing output files are\n"
        "never overwritten. The archive does not store the shape: keep it for decode.\n",
        program);
}

bool parse_dimension(const char* text, std::uint32_t* result) {
    if (text == nullptr || *text == '\0')
        return false;
    std::uint32_t value = 0;
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9')
            return false;
        const unsigned digit = static_cast<unsigned>(*p - '0');
        if (value > (std::numeric_limits<std::uint32_t>::max() - digit) / 10u)
            return false;
        value = value * 10u + digit;
    }
    if (value == 0)
        return false;
    *result = value;
    return true;
}

bool parse_bound(const char* text, double* result) {
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE || !std::isfinite(value) || !(value > 0))
        return false;
    *result = value;
    return true;
}

struct Buffer {
    void* data = nullptr;
    ~Buffer() { std::free(data); }
    bool allocate(std::size_t bytes) {
        data = std::malloc(bytes);
        if (data != nullptr)
            return true;
        std::fprintf(stderr, "Cannot allocate %zu bytes.\n", bytes);
        return false;
    }
};

struct InputFile {
    FILE* file = nullptr;
    ~InputFile() { if (file != nullptr) std::fclose(file); }
    bool open(const char* path) {
        file = std::fopen(path, "rb");
        if (file != nullptr)
            return true;
        std::fprintf(stderr, "Cannot open input '%s': %s\n", path, std::strerror(errno));
        return false;
    }
    bool size(std::size_t* bytes, const char* path) {
        if (std::fseek(file, 0, SEEK_END) != 0) {
            std::fprintf(stderr, "Cannot seek input '%s': %s\n", path, std::strerror(errno));
            return false;
        }
        const long length = std::ftell(file);
        if (length < 0) {
            std::fprintf(stderr, "Cannot measure input '%s': %s\n", path, std::strerror(errno));
            return false;
        }
        if (static_cast<unsigned long long>(length) > std::numeric_limits<std::size_t>::max()) {
            std::fprintf(stderr, "Input '%s' exceeds the host address space.\n", path);
            return false;
        }
        if (std::fseek(file, 0, SEEK_SET) != 0) {
            std::fprintf(stderr, "Cannot rewind input '%s': %s\n", path, std::strerror(errno));
            return false;
        }
        *bytes = static_cast<std::size_t>(length);
        return true;
    }
    bool read(void* data, std::size_t bytes, const char* path) {
        if (std::fread(data, 1, bytes, file) != bytes) {
            if (std::ferror(file))
                std::fprintf(stderr, "Cannot read input '%s': %s\n", path, std::strerror(errno));
            else
                std::fprintf(stderr, "Input '%s' is shorter than the required %zu bytes.\n", path, bytes);
            return false;
        }
        const int extra = std::fgetc(file);
        if (extra != EOF) {
            std::fprintf(stderr, "Input '%s' is longer than the required %zu bytes.\n", path, bytes);
            return false;
        }
        if (std::ferror(file)) {
            std::fprintf(stderr, "Cannot finish reading input '%s': %s\n", path, std::strerror(errno));
            return false;
        }
        FILE* closing = file;
        file = nullptr;
        if (std::fclose(closing) != 0) {
            std::fprintf(stderr, "Cannot close input '%s': %s\n", path, std::strerror(errno));
            return false;
        }
        return true;
    }
};

struct Workspace {
    waltz_cpu_workspace* value = waltz_cpu_create();
    ~Workspace() { waltz_cpu_destroy(value); }
};

bool write_exclusive(const char* path, const void* data, std::size_t bytes) {
    // C11 'x' creation atomically refuses an existing file (including symlinks).
    FILE* file = std::fopen(path, "wbx");
    if (file == nullptr) {
        std::fprintf(stderr, "Cannot create output '%s' without overwriting: %s\n",
                     path, std::strerror(errno));
        return false;
    }
    bool success = std::fwrite(data, 1, bytes, file) == bytes;
    int write_error = success ? 0 : errno;
    if (success && std::fflush(file) != 0) {
        success = false;
        write_error = errno;
    }
    if (std::fclose(file) != 0 && success) {
        success = false;
        write_error = errno;
    }
    if (!success) {
        std::fprintf(stderr, "Cannot finish output '%s': %s\n", path,
                     write_error != 0 ? std::strerror(write_error) : "short write");
        // This exact path was created exclusively by this invocation.
        if (std::remove(path) != 0)
            std::fprintf(stderr, "Warning: incomplete output '%s' could not be removed: %s\n",
                         path, std::strerror(errno));
    }
    return success;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        usage(stdout, argv[0]);
        return EXIT_SUCCESS;
    }
    if (argc != 9 && argc != 10) {
        usage(stderr, argv[0]);
        return 2;
    }
    const bool compress = std::strcmp(argv[1], "c") == 0;
    if (!compress && std::strcmp(argv[1], "d") != 0) {
        std::fprintf(stderr, "Operation must be 'c' or 'd'.\n");
        return 2;
    }
    waltz_cpu_config config{};
    if (std::strcmp(argv[2], "f32") == 0)
        config.type = WALTZ_CPU_F32;
    else if (std::strcmp(argv[2], "f64") == 0)
        config.type = WALTZ_CPU_F64;
    else {
        std::fprintf(stderr, "Type must be 'f32' or 'f64'.\n");
        return 2;
    }
    if (!parse_dimension(argv[3], &config.nx) || !parse_dimension(argv[4], &config.ny) ||
        !parse_dimension(argv[5], &config.nz)) {
        std::fprintf(stderr, "Dimensions must be positive uint32 decimal integers.\n");
        return 2;
    }
    const std::uint64_t max_elements = std::numeric_limits<std::uint32_t>::max() /
                                       static_cast<unsigned>(config.type);
    if (config.nx > max_elements / config.ny ||
        std::uint64_t(config.nx) * config.ny > max_elements / config.nz) {
        std::fprintf(stderr, "Shape exceeds the 32-bit native archive section limit.\n");
        return 2;
    }
    const std::uint64_t elements = std::uint64_t(config.nx) * config.ny * config.nz;
    const std::uint64_t raw64 = elements * static_cast<unsigned>(config.type);
    if (raw64 > std::numeric_limits<std::size_t>::max()) {
        std::fprintf(stderr, "Shape exceeds the host address space.\n");
        return 2;
    }
    const std::size_t raw_bytes = static_cast<std::size_t>(raw64);
    if (!parse_bound(argv[6], &config.absolute_error)) {
        std::fprintf(stderr, "Bound must be a positive finite number.\n");
        return 2;
    }
    const char* const backend = argc == 10 ? argv[9] : "auto";
    if (std::strcmp(backend, "scalar") == 0)
        config.backend = WALTZ_CPU_SCALAR;
    else if (std::strcmp(backend, "avx2") == 0)
        config.backend = WALTZ_CPU_AVX2;
    else if (std::strcmp(backend, "auto") == 0)
        config.backend = WALTZ_CPU_AUTO;
    else {
        std::fprintf(stderr, "Backend must be 'scalar', 'avx2', or 'auto'.\n");
        return 2;
    }
    if (config.backend == WALTZ_CPU_AVX2 && !waltz_cpu_has_avx2()) {
        std::fprintf(stderr, "AVX2/FMA backend is not available on this CPU.\n");
        return 2;
    }
    const std::size_t archive_capacity = waltz_cpu_max_compressed_bytes(&config);
    if (archive_capacity == 0) {
        std::fprintf(stderr, "Configuration is unsupported or its archive size overflows.\n");
        return 2;
    }
    const char* const input_path = argv[7];
    const char* const output_path = argv[8];
    InputFile input;
    if (!input.open(input_path))
        return EXIT_FAILURE;
    std::size_t input_bytes = 0;
    if (!input.size(&input_bytes, input_path))
        return EXIT_FAILURE;
    if (compress && input_bytes != raw_bytes) {
        std::fprintf(stderr, "Raw input '%s' has %zu bytes; shape/type require exactly %zu.\n",
                     input_path, input_bytes, raw_bytes);
        return EXIT_FAILURE;
    }
    if (!compress && (input_bytes < 40u || input_bytes > archive_capacity)) {
        std::fprintf(stderr, "Archive '%s' has an invalid size for the supplied shape/type.\n", input_path);
        return EXIT_FAILURE;
    }
    Buffer source, destination;
    if (!source.allocate(input_bytes) || !input.read(source.data, input_bytes, input_path))
        return EXIT_FAILURE;
    if (!destination.allocate(compress ? archive_capacity : raw_bytes))
        return EXIT_FAILURE;
    Workspace workspace;
    if (workspace.value == nullptr) {
        std::fprintf(stderr, "Cannot create the CPU codec workspace.\n");
        return EXIT_FAILURE;
    }
    std::size_t output_bytes = raw_bytes;
    const int status = compress
        ? waltz_cpu_compress(workspace.value, &config, source.data, destination.data,
                             archive_capacity, &output_bytes)
        : waltz_cpu_decompress(workspace.value, &config, source.data, input_bytes,
                               destination.data, raw_bytes);
    if (status != 0) {
        std::fprintf(stderr, "%s failed: %s\n", compress ? "Compression" : "Decompression",
                     waltz_cpu_last_error(workspace.value));
        return EXIT_FAILURE;
    }
    if (!write_exclusive(output_path, destination.data, output_bytes))
        return EXIT_FAILURE;
    std::printf("%s %zu -> %zu bytes (%s); wrote '%s'.\n",
                compress ? "Compressed" : "Decompressed", input_bytes, output_bytes,
                backend, output_path);
    return EXIT_SUCCESS;
}
