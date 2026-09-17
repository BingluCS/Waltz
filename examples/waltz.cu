#include "utils/Blob.hpp"
#include "utils/io.hpp"
#include "waltz.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <cuda_runtime.h>

[[noreturn]] inline void usage(int status = EXIT_FAILURE) {
    std::printf("Usage:\n"
                "  Compress:   waltz [-f|-d] -i input -z compressed -3 nx ny nz -M REL|ABS bound [-o output]\n"
                "  Decompress: waltz [-f|-d] -z compressed -o output -3 nx ny nz\n"
                "              (-i compressed -o output also works without -M)\n"
                "\nOptions:\n"
                "  -h                  Print this help and exit.\n"
                "  -f / -d             float32 (default) / float64; -d selects the type, not decompression.\n"
                "  -i input            Raw input when compressing; compressed input when decoding without -M.\n"
                "  -z compressed       Compressed output when compressing; compressed input when decoding.\n"
                "  -o output           Write reconstruction; with compression, also verify against the input.\n"
                "  -3 nx ny nz         Original dimensions; required for both compression and decompression.\n"
                "  -M REL|ABS bound    Compress with REL * (max - min), or an absolute error bound.\n"
                "  Compressed files do not store dimensions or data type; supply the same values on decode.\n"
                "\nTiming:\n"
                "  WALTZ_WARMUP=1: run each requested API once before the measured call.\n"
                "  WALTZ_REPORT_CE2E=1: print API wall time (excluding REL scan, file I/O and external transfers).\n"
                "\nExamples:\n"
                "  waltz -f -i field.f32 -z field.waltz -3 512 512 512 -M REL 1e-3\n"
                "  waltz -f -z field.waltz -o field.out.f32 -3 512 512 512\n");
    std::exit(status);
}

template <class T>
void Waltz_compress(const char* inPath, const char* cmpPath, WALTZ::Config conf) {
    std::unique_ptr<T[]> oridata(WALTZ::readfile<T>(inPath, conf.num));

    T* d_oridata;
    char* d_cmpdata;
    cudaMalloc(reinterpret_cast<void**>(&d_oridata), conf.num * sizeof(T));
    cudaMemcpy(d_oridata, oridata.get(), conf.num * sizeof(T), cudaMemcpyHostToDevice);
    const size_t cmp_capacity = WALTZ::compressed_buffer_capacity<T>(conf.num);
    cudaMalloc(reinterpret_cast<void**>(&d_cmpdata), cmp_capacity);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Resolve REL once, outside both warm-up and compression API timing.
    if (conf.errorBoundMode == WALTZ::EB_REL) {
        double dmin = 0.0, dmax = 0.0;
        WALTZ::d_extrema<T>(d_oridata, static_cast<uint64_t>(conf.num), dmin, dmax, stream);
        conf.absErrorBound = conf.relErrorBound * (dmax - dmin);
        conf.errorBoundMode = WALTZ::EB_ABS;
    }
    const char* warmup = std::getenv("WALTZ_WARMUP");
    if (warmup != nullptr && std::atoi(warmup) != 0) {
        WALTZ::Waltz_report_enabled = false;
        WALTZ::d_compress<T>(conf, d_oridata, d_cmpdata, cmpPath, stream);
        WALTZ::Waltz_report_enabled = true;
    }
    cudaStreamSynchronize(stream);
    const WALTZ::WaltzE2ETimer ce2e_timer;
    size_t outSize = WALTZ::d_compress<T>(conf, d_oridata, d_cmpdata, cmpPath, stream);
    ce2e_timer.report("compress", conf.num * sizeof(T), stream);

    if (outSize > 0) {
        std::unique_ptr<char[]> h_cmpdata(new char[outSize]);
        cudaMemcpy(h_cmpdata.get(), d_cmpdata, outSize, cudaMemcpyDeviceToHost);
        WALTZ::writefile(cmpPath, h_cmpdata.get(), outSize);
    }
    cudaFree(d_cmpdata);
    cudaFree(d_oridata);
    cudaStreamDestroy(stream);
}

template <class T>
void Waltz_decompress(const char* cmpPath, const char* decPath, const WALTZ::Config& conf,
                      const char* originalPath = nullptr) {
    std::ifstream file(cmpPath, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < static_cast<std::streamoff>(sizeof(WALTZ::WaltzBlobHeader)))
        throw std::runtime_error("Cannot read compressed file or header is incomplete");
    const size_t cmp_size = static_cast<size_t>(file.tellg());
    std::unique_ptr<char[]> h_cmpdata(new char[cmp_size]);
    file.seekg(0);
    if (!file.read(h_cmpdata.get(), cmp_size))
        throw std::runtime_error("Cannot read compressed file");

    char* d_cmpdata = nullptr;
    T* d_decdata = nullptr;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMalloc(reinterpret_cast<void**>(&d_cmpdata), cmp_size);
    cudaMalloc(reinterpret_cast<void**>(&d_decdata), conf.num * sizeof(T));
    cudaMemcpyAsync(d_cmpdata, h_cmpdata.get(), cmp_size, cudaMemcpyHostToDevice, stream);
    const char* warmup = std::getenv("WALTZ_WARMUP");
    if (warmup != nullptr && std::atoi(warmup) != 0) {
        WALTZ::Waltz_report_enabled = false;
        WALTZ::d_decompress<T>(conf, d_cmpdata, d_decdata, nullptr, stream);
        WALTZ::Waltz_report_enabled = true;
    }
    cudaStreamSynchronize(stream);
    const WALTZ::WaltzE2ETimer ce2e_timer;
    const size_t bytes = WALTZ::d_decompress<T>(conf, d_cmpdata, d_decdata, nullptr, stream);
    ce2e_timer.report("decompress", conf.num * sizeof(T), stream);
    if (bytes != conf.num * sizeof(T))
        throw std::runtime_error("Unexpected decompressed size");
    std::unique_ptr<T[]> output(new T[conf.num]);
    cudaMemcpy(output.get(), d_decdata, bytes, cudaMemcpyDeviceToHost);
    WALTZ::writefile(decPath, output.get(), conf.num);
    cudaFree(d_decdata);
    cudaFree(d_cmpdata);
    cudaStreamDestroy(stream);
    // Optional host-side verification, outside the kernel and API e2e timers.
    if (originalPath != nullptr) {
        std::unique_ptr<T[]> original(WALTZ::readfile<T>(originalPath, conf.num));
        statistic<T>(original.get(), output.get(), conf.num);
    }
}

int main(int argc, char* argv[]) try {
    bool compression = false;
    bool decompression = false;

    int dataType = WALTZ_FLOAT;

    char* inPath = nullptr;
    char* cmpPath = nullptr;
    char* decPath = nullptr;

    char* errBoundMode = nullptr;
    char* errBound = nullptr;

    size_t r4 = 0;
    size_t r3 = 0;
    size_t r2 = 0;
    size_t r1 = 0;

    int i = 0;
    if (argc == 1)
        usage();
    int width = -1;

    for (i = 1; i < argc; i++) {
        if (std::strlen(argv[i]) != 2 || argv[i][0] != '-') {
            usage();
        }

        switch (argv[i][1]) {
        case 'h':
            usage(EXIT_SUCCESS);
            break;
        case 'z':
            compression = true;
            if (i + 1 < argc) {
                cmpPath = argv[i + 1];
                if (cmpPath[0] != '-')
                    i++;
                else
                    cmpPath = nullptr;
            }
            break;
        case 'o':
            decompression = true;
            if (i + 1 < argc) {
                decPath = argv[i + 1];
                if (decPath[0] != '-')
                    i++;
                else
                    decPath = nullptr;
            }
            break;
        case 'f':
            dataType = WALTZ_FLOAT;
            break;
        case 'd':
            dataType = WALTZ_DOUBLE;
            break;
        case 'I':
            if (++i == argc || sscanf(argv[i], "%d", &width) != 1) {
                usage();
            }
            if (width == 32) {
                dataType = WALTZ_INT32;
            } else if (width == 64) {
                dataType = WALTZ_INT64;
            } else {
                usage();
            }
            break;
        case 'i':
            if (++i == argc)
                usage();
            inPath = argv[i];
            break;
        case '1':
            if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1)
                usage();
            break;
        case '2':
            if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1 || ++i == argc ||
                sscanf(argv[i], "%zu", &r2) != 1)
                usage();
            break;
        case '3':
            if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1 || ++i == argc ||
                sscanf(argv[i], "%zu", &r2) != 1 || ++i == argc || sscanf(argv[i], "%zu", &r3) != 1)
                usage();
            break;
        case '4':
            if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1 || ++i == argc ||
                sscanf(argv[i], "%zu", &r2) != 1 || ++i == argc ||
                sscanf(argv[i], "%zu", &r3) != 1 || ++i == argc || sscanf(argv[i], "%zu", &r4) != 1)
                usage();
            break;
        case 'M':
            if (++i == argc)
                usage();
            errBoundMode = argv[i];
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                errBound = argv[++i];
            }
            break;
        default:
            usage();
            break;
        }
    }

    if ((inPath == nullptr) && (cmpPath == nullptr)) {
        printf("Error: you need to specify either a raw binary data file or a compressed data file "
               "as input\n");
        usage();
        exit(0);
    }

    char cmpPathTmp[1024];
    if (inPath != nullptr && cmpPath == nullptr && decPath != nullptr && errBoundMode != nullptr) {
        compression = true;
        decompression = true;
        snprintf(cmpPathTmp, 1024, "%s.waltz.tmp", inPath);
        cmpPath = cmpPathTmp;
    }
    if (inPath == nullptr || errBoundMode == nullptr) {
        compression = false;
    }
    if (!compression && !decompression) {
        usage();
        exit(0);
    }
    if (decompression && decPath == nullptr)
        usage();
    if (compression && cmpPath == nullptr)
        usage();
    if (!compression && cmpPath == nullptr)
        cmpPath = inPath;

    WALTZ::Config conf;
    if (r2 == 0) {
        conf = WALTZ::Config(r1);
    } else if (r3 == 0) {
        conf = WALTZ::Config(r1, r2);
    } else if (r4 == 0) {
        conf = WALTZ::Config(r1, r2, r3);
    } else {
        conf = WALTZ::Config(r1, r2, r3, r4);
    }

    if (conf.ndims != 3)
        throw std::invalid_argument("Specify the original 3D dimensions with -3 nx ny nz");

    if (errBoundMode != nullptr) {

        WALTZ::match_enum(errBoundMode, WALTZ::EB_MAP, conf.errorBoundMode);
        if (conf.errorBoundMode == WALTZ::EB_ABS) {
            if (errBound != nullptr) {
                conf.absErrorBound = atof(errBound);
            }
        } else if (conf.errorBoundMode == WALTZ::EB_REL) {
            if (errBound != nullptr) {
                conf.relErrorBound = atof(errBound);
            }
        } else if (conf.errorBoundMode == WALTZ::EB_PSNR) {
            if (errBound != nullptr) {
                conf.psnrErrorBound = atof(errBound);
            }
        } else {
            printf("Error: wrong error bound mode setting by using the option '-M'\n");
            usage();
            exit(0);
        }
    }

    if (compression) {
        if (dataType == WALTZ_FLOAT) {
            Waltz_compress<float>(inPath, cmpPath, conf);
        } else if (dataType == WALTZ_DOUBLE) {
            Waltz_compress<double>(inPath, cmpPath, conf);
        } else {
            printf("Error: data type not supported \n");
            usage();
            exit(0);
        }
    }
    if (decompression) {
        if (dataType == WALTZ_FLOAT) {
            Waltz_decompress<float>(cmpPath, decPath, conf, compression ? inPath : nullptr);
        } else if (dataType == WALTZ_DOUBLE) {
            Waltz_decompress<double>(cmpPath, decPath, conf, compression ? inPath : nullptr);
        } else {
            std::fprintf(stderr, "Error: data type not supported\n");
            usage();
        }
    }

    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "Error: %s\n", error.what());
    return EXIT_FAILURE;
}
