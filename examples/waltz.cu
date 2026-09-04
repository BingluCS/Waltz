#include "utils/Statistics.hpp"
#include "waltz.hpp"

#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>

inline void usage(FILE* out) {
    std::fprintf(out,
                 "Usage:\n"
                 "  waltz -f|-d -i INPUT -z ARCHIVE [-o RECONSTRUCTED] "
                 "-3 NX NY NZ -M ABS|REL ERROR\n\n"
                 "Options:\n"
                 "  -f              input is float32\n"
                 "  -d              input is float64\n"
                 "  -i PATH         raw input field\n"
                 "  -z PATH         write the compressed Waltz blob\n"
                 "  -o PATH         also decompress, verify, and write reconstruction\n"
                 "  -3 NX NY NZ     dimensions in X, Y, Z order\n"
                 "  -M ABS VALUE    absolute point-wise error bound\n"
                 "  -M REL VALUE    value-range-relative point-wise error bound\n"
                 "  -h              show this help\n");
}

template <class T>
void compress(char* inPath, char* cmpPath, char* decPath, WALTZ::Config conf) {
    T* oridata = WALTZ::readfile<T>(inPath, conf.num);

    T* d_oridata;
    char* d_cmpdata;
    WALTZ::check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_oridata), conf.num * sizeof(T)),
                      "cudaMalloc input");
    WALTZ::check_cuda(
        cudaMemcpy(d_oridata, oridata, conf.num * sizeof(T), cudaMemcpyHostToDevice),
        "copy input H2D");
    const size_t cmp_capacity = WALTZ::compressed_buffer_capacity<T>(conf.num);
    WALTZ::check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_cmpdata), cmp_capacity),
                      "cudaMalloc compressed output");
    cudaStream_t stream;
    WALTZ::check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    size_t outSize = WALTZ::d_compress<T>(conf, d_oridata, d_cmpdata, cmpPath, stream);

    if (outSize > 0) {
        std::vector<unsigned char> blob(outSize);
        WALTZ::check_cuda(cudaMemcpy(blob.data(), d_cmpdata, outSize, cudaMemcpyDeviceToHost),
                          "copy compressed blob D2H");
        std::ofstream output(cmpPath, std::ios::binary);
        if (!output)
            throw std::runtime_error(std::string("cannot open compressed output: ") + cmpPath);
        output.write(reinterpret_cast<const char*>(blob.data()),
                     static_cast<std::streamsize>(blob.size()));
        if (!output)
            throw std::runtime_error(std::string("cannot write compressed output: ") + cmpPath);
    }

    // ---- decompress the just-produced blob and verify the point-wise bound ----
    if (outSize > 0 && decPath != nullptr) {
        T* d_decdata = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&d_decdata), conf.num * sizeof(T));
        size_t decSize = WALTZ::d_decompress<T>(conf, d_cmpdata, d_decdata, decPath, stream);
        (void)decSize;
        T* dec = new T[conf.num];
        cudaMemcpy(dec, d_decdata, conf.num * sizeof(T), cudaMemcpyDeviceToHost);
        printf("[WALTZ] DECOMPRESS verify: blob %zu B (CR=%.2fx), decompressed vs original:\n",
               outSize,
               conf.num * 1.0 * sizeof(T) / outSize);
        statistic<T>(oridata, dec, conf.num);
        delete[] dec;
        cudaFree(d_decdata);
    }

    WALTZ::check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    WALTZ::check_cuda(cudaFree(d_cmpdata), "cudaFree compressed output");
    WALTZ::check_cuda(cudaFree(d_oridata), "cudaFree input");
    delete[] oridata;
}

int main(int argc, char* argv[]) {
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
    if (argc == 1) {
        usage(stderr);
        return 2;
    }
    int width = -1;

    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-' || argv[i][2]) {
            usage(stderr);
            return 2;
        }

        switch (argv[i][1]) {
        case 'h':
            usage(stdout);
            return 0;
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
                usage(stderr);
                return 2;
            }
            if (width == 32) {
                dataType = WALTZ_INT32;
            } else if (width == 64) {
                dataType = WALTZ_INT64;
            } else {
                usage(stderr);
                return 2;
            }
            break;
        case 'i':
            if (++i == argc) {
                usage(stderr);
                return 2;
            }
            inPath = argv[i];
            break;
        case '1':
            if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1) {
                usage(stderr);
                return 2;
            }
            break;
        case '2':
            if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1 || ++i == argc ||
                sscanf(argv[i], "%zu", &r2) != 1) {
                usage(stderr);
                return 2;
            }
            break;
        case '3':
            if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1 || ++i == argc ||
                sscanf(argv[i], "%zu", &r2) != 1 || ++i == argc ||
                sscanf(argv[i], "%zu", &r3) != 1) {
                usage(stderr);
                return 2;
            }
            break;
        case '4':
            if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1 || ++i == argc ||
                sscanf(argv[i], "%zu", &r2) != 1 || ++i == argc ||
                sscanf(argv[i], "%zu", &r3) != 1 || ++i == argc ||
                sscanf(argv[i], "%zu", &r4) != 1) {
                usage(stderr);
                return 2;
            }
            break;
        case 'M':
            if (++i == argc) {
                usage(stderr);
                return 2;
            }
            errBoundMode = argv[i];
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                errBound = argv[++i];
            }
            break;
        default:
            usage(stderr);
            return 2;
        }
    }

    if ((inPath == nullptr) && (cmpPath == nullptr)) {
        printf("Error: you need to specify either a raw binary data file or a compressed data file "
               "as input\n");
        usage(stderr);
        return 2;
    }

    char cmpPathTmp[1024];
    if (inPath != nullptr && cmpPath == nullptr && decPath != nullptr) {
        compression = true;
        decompression = true;
        snprintf(cmpPathTmp, 1024, "%s.sz.tmp", inPath);
        cmpPath = cmpPathTmp;
    }
    if (inPath == nullptr || errBoundMode == nullptr) {
        compression = false;
    }
    if (!compression || inPath == nullptr || cmpPath == nullptr ||
        (decompression && decPath == nullptr) || r1 == 0 || r2 == 0 || r3 == 0 || r4 != 0 ||
        errBound == nullptr) {
        usage(stderr);
        return 2;
    }

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
            usage(stderr);
            return 2;
        }
    }

    if (compression) {
        if (dataType == WALTZ_FLOAT) {
            compress<float>(inPath, cmpPath, decPath, conf);
        } else if (dataType == WALTZ_DOUBLE) {
            compress<double>(inPath, cmpPath, decPath, conf);
        } else {
            printf("Error: data type not supported \n");
            usage(stderr);
            return 2;
        }
    }

    return 0;
}
