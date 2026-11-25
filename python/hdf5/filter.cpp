// Waltz private H5Z filter. Client data comes from pywaltz.hdf5.create_dataset;
// the filter itself needs no hid_t handles or HDF5 library calls.
#include <H5PLextern.h>
#include "waltz_cpu.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

namespace {
constexpr H5Z_filter_t filter_id = 61197;  // Private/unregistered, not Falcons' ID.
static_assert(sizeof(unsigned) == 4 && sizeof(double) == 8 &&
              std::numeric_limits<double>::is_iec559);

size_t filter(unsigned flags, size_t count, const unsigned cd[], size_t bytes,
              size_t* buffer_size, void** buffer) {
    const uint16_t one = 1;
    if (*reinterpret_cast<const uint8_t*>(&one) != 1 || count != 6 || !cd ||
        !buffer || !*buffer || !buffer_size || !bytes || bytes > *buffer_size ||
        (cd[3] != 4 && cd[3] != 8)) return 0;
    const uint64_t bits = uint64_t(cd[4]) | (uint64_t(cd[5]) << 32);
    double bound;
    std::memcpy(&bound, &bits, sizeof(bound));
    const waltz_cpu_config cfg{cd[0], cd[1], cd[2], static_cast<waltz_cpu_type>(cd[3]),
                               bound, WALTZ_CPU_AUTO};
    // The codec checks dimensions, bound, multiplication and archive size limits.
    const size_t max_archive = waltz_cpu_max_compressed_bytes(&cfg);
    if (!max_archive) return 0;
    const size_t raw = size_t(cfg.nx) * cfg.ny * cfg.nz * cd[3];
    const bool decode = (flags & H5Z_FLAG_REVERSE) != 0;
    if (decode ? (bytes < 32 || bytes > max_archive) : bytes != raw) return 0;

    size_t capacity = decode ? raw : max_archive;
    void* out = std::malloc(capacity);
    if (!out) return 0;
    std::unique_ptr<waltz_cpu_workspace, decltype(&waltz_cpu_destroy)>
        ws(waltz_cpu_create(), waltz_cpu_destroy);
    size_t written = raw;
    if (!ws || (decode ? waltz_cpu_decompress(ws.get(), &cfg, *buffer, bytes, out, raw)
                       : waltz_cpu_compress(ws.get(), &cfg, *buffer, out, capacity, &written))) {
        std::free(out);
        return 0;  // H5Z requires the original buffer to remain untouched on failure.
    }
    if (written < capacity) {
        if (void* smaller = std::realloc(out, written)) { out = smaller; capacity = written; }
    }
    std::free(*buffer);
    *buffer = out;
    *buffer_size = capacity;
    return written;
}
const H5Z_class2_t descriptor{H5Z_CLASS_T_VERS, filter_id, 1, 1, "Waltz canonical",
                               nullptr, nullptr, filter};
}

extern "C" H5PL_type_t H5PLget_plugin_type() { return H5PL_TYPE_FILTER; }
extern "C" const void* H5PLget_plugin_info() { return &descriptor; }
