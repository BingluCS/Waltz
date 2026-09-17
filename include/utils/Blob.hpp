#pragma once

#include <cstdint>

namespace WALTZ {

// Current compressed-stream header; 8-byte-aligned sections follow:
// WZP magnitude/sign stream, PWE indices/errors, quantization outlier indices/values.
// Dimensions and element count come from Config; the caller supplies the data type.
struct WaltzBlobHeader {
    uint8_t use_dyadic; // 0=plane, 1=dyadic
    uint8_t levels_xy, levels_z;
    uint8_t reorder_bz;

    double q;
    uint64_t mag_comp_bytes;
    uint32_t n_pwe, n_qo;
};
static_assert(sizeof(WaltzBlobHeader) == 32, "Waltz blob header layout must remain 32 bytes");

} // namespace WALTZ
