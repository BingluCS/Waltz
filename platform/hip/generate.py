#!/usr/bin/env python3
"""Materialize the HIP subset of the CUDA backend, without modifying CUDA."""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)


def edit(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f'CUDA source drift: {old!r}')
    return text.replace(old, new)


def shuffles(text):
    return re.sub(r'__(shfl(?:_up|_down|_xor)?)_sync\(0x[fF]{8}[uU]?,\s*([^,\n]+),\s*([^() ,\n]+)\)',
                  r'__\1(\2, \3, 32)', text)


def header(text):
    return text.replace('#include <cuda_runtime.h>', '#include <hip/hip_runtime.h>')


warp = header((root / 'include/transformers/warp97.cuh').read_text())
warp = shuffles(warp)
assert '_sync(' not in warp
warp = warp.replace('__fmaf_rn(a,b,c)', 'fmaf(a,b,c)').replace('__fma_rn(a,b,c)', 'fma(a,b,c)')
warp = edit(warp, '''    #pragma unroll
    for (int j=0; j<8; ++j) { e[j]*=Constants<T>::IK; o[j]*=-Constants<T>::K; }
    #pragma unroll
    for (int j=1; j<8; ++j) e[j]=madd(-Constants<T>::D,o[j-1]+o[j],e[j]);''',
'''    const T first=o[0],last=o[7];
    #pragma unroll
    for (int j=0; j<8; ++j) { e[j]*=Constants<T>::IK; o[j]*=-Constants<T>::K; }
    #pragma unroll
    for (int j=1; j<8; ++j) {
        const T sum=j==1?madd(first,-Constants<T>::K,o[1]):
                    j==7?madd(last,-Constants<T>::K,o[6]):o[j-1]+o[j];
        e[j]=madd(-Constants<T>::D,sum,e[j]);
    }''')
(out / 'warp97_hip.cuh').write_text(warp)

halo = (root / 'include/transformers/halo3d.cuh').read_text().split('// XY transforms of halo-expanded')[0]
halo = edit(halo, '#include "warp97.cuh"', '#include "warp97_hip.cuh"')
# The HIP subset uses only fixed halo tiles: x overread <=59, y/z <=19.
# Supported volumes fit in uint32, so the GPU element offsets do too.
halo = edit(halo, 'if(n>=128) {', 'if(n>=64) {')
halo = edit(halo, 'const int r=periodic_mirror(p,n); return r/2+(r%2?(n+1)/2:0);',
            'const int r=mirror(p,n); return r/2+((r&1)?(n+1)/2:0);')
halo = edit(halo, '''    #pragma unroll
    for(int j=0;j<P;++j) {e[j]*=Constants<T>::IK; o[j]*=-Constants<T>::K;}
    #pragma unroll
    for(int j=1;j<P;++j) e[j]=madd(-Constants<T>::D,o[j-1]+o[j],e[j]);''',
'''    const T first=o[0],last=o[P-1];
    #pragma unroll
    for(int j=0;j<P;++j) {e[j]*=Constants<T>::IK; o[j]*=-Constants<T>::K;}
    #pragma unroll
    for(int j=1;j<P;++j) {
        const T sum=j==1?madd(first,-Constants<T>::K,o[1]):
                    j==P-1?madd(last,-Constants<T>::K,o[P-2]):o[j-1]+o[j];
        e[j]=madd(-Constants<T>::D,sum,e[j]);
    }''')
(out / 'halo3d_hip.cuh').write_text(halo + '} // namespace halo3d\n')

fused = (root / 'include/transformers/all_halo.cuh').read_text()
fused = edit(fused, 'return (size_t(z)*full.y+y)*full.x+x;',
             'return (uint32_t(z)*full.y+uint32_t(y))*full.x+uint32_t(x);')
begin = fused.index('namespace WALTZ::allhalo {')
end = fused.index('template<class T,bool Regular', begin)
# Both native and batched HIP transforms use the split XY/Z kernels.
kernels = fused.index('template<class T,bool Inverse,class Emit,bool Batch=false> __global__ void xy', end)
stop = fused.index('template<class T,bool Forward,class Emit> __global__ void identity', kernels)
(out / 'fused_hip.cuh').write_text('#pragma once\n#include "halo3d_hip.cuh"\n'
                                 + fused[begin:end] + fused[kernels:stop] + '} // namespace\n')
(out / 'reorder_hip.cuh').write_text(header((root / 'include/lossless/reorder.hpp').read_text()))

wzp = (root / 'src/lossless/wzp.cu').read_text().split('\nvoid release()')[0]
wzp = header(wzp)
wzp = edit(wzp, '#include <cuda/atomic>\n', '')
wzp = edit(wzp, '#include "lossless/wzp.hpp"\n#include "utils/Config.hpp"\n', '')
wzp = edit(wzp, '    byte* __restrict__ cmpdata, unsigned int* __restrict__ group_done,\n'
                '    unsigned long long* __restrict__ group_carry) {',
'''    unsigned int* __restrict__ next_chunk) {''')
wzp = edit(wzp, '__global__\n__launch_bounds__(TPB, 2048 / TPB)\nvoid encode_kernel',
                'template<bool PackedCodes=false>\n__global__\n__launch_bounds__(TPB, 2048 / TPB)\nvoid encode_kernel')
wzp = edit(wzp, 'if ((sign_bitmap[global >> 5u] >> (global & 31u)) & 1u)',
                '''const unsigned c=mag[i];
                    if constexpr(PackedCodes) {
                        const unsigned m=c&32767u;
                        reinterpret_cast<uint16_t*>(raw)[i]=m==32767u?
                            reinterpret_cast<const uint16_t*>(sign_bitmap)[global]:uint16_t(m);
                    }
                    if (PackedCodes ? bool(c>>15) : ((sign_bitmap[global >> 5u] >> (global & 31u)) & 1u) != 0)''')
wzp = edit(wzp,'                encoded = input + base;', '''                if constexpr(PackedCodes){
                    int bytes=int(raw_bytes);
                    // RAW slots contain magnitudes, not the sign-tagged staging words.
                    d_iBIT_2(bytes,transformed,raw,temp);
                    __syncthreads();
                    encoded=raw;
                }else encoded=input+base;''')
beg = wzp.index('__device__ unsigned int payload_size;')
end = wzp.index('__device__ __forceinline__ uint32_t load_meta24', beg)
wzp = wzp[:beg] + '''__device__ __forceinline__ uint32_t ballot32(bool pred) {
    return uint32_t(__ballot(pred) >> (threadIdx.x & 32u));
}

''' + wzp[end:]
wzp = edit(wzp, '''    __shared__ uint32_t block_nnz;
    __shared__ bool pack_ready;
    if (threadIdx.x == 0) block_nnz = 0U;''', '')
wzp = edit(wzp, 'atomicAdd(&chunk_counter, 1ull)', 'atomicAdd(next_chunk, 1u)')
wzp = edit(wzp, '''        if (chunk >= chunks) {
            if (threadIdx.x == 0 && block_nnz != 0U)
                atomicAdd(&nnz_counter, block_nnz);
            return;
        }''', '        if (chunk >= chunks) return;')
wzp = edit(wzp, '''            if (threadIdx.x == 0)
                block_nnz += chunk_nnz;
''', '')
beg = wzp.index('        if (!zero)\n            __syncthreads();\n        // Publish')
end = wzp.index('\ntemplate <int DirectLayout', beg)
wzp = wzp[:beg] + '        __syncthreads();\n    }\n}\n' + wzp[end:]
wzp = shuffles(wzp)
# A wave64 lane expands 16 consecutive sparse signs with one block prefix.
# The writer still consumes exactly the original halfword sign bitmap.
decode = wzp.index('template <int DirectLayout')
helper = ((root / 'platform/hip/sign_groups.cuh').read_text() + '\n' +
          (root / 'platform/hip/rze_full.cuh').read_text())
begin = wzp.index('        const uint32_t lane = threadIdx.x & 31U;', wzp.index('const byte* packed_sign = source + stored_bytes;',decode))
end = wzp.index('        if constexpr (DirectLayout != 0)', begin)
wzp = (wzp[:decode] + helper + '\n' + wzp[decode:begin] +
       '        expand_sign_groups(mag, elements, packed_sign, sign_bytes, temp);\n'
       '        auto* const sign_words = reinterpret_cast<uint32_t*>(temp);\n' + wzp[end:])
wzp = edit(wzp, '        d_iRZE_1(compressed_bytes, encoded, expanded, temp);',
                '        if(raw_bytes==CS && stored_bytes>=4){\n'
                '            rze_full(encoded,stored_bytes,expanded,temp);compressed_bytes=CS;\n'
                '        }else d_iRZE_1(compressed_bytes, encoded, expanded, temp);')
assert not re.search(r'__shfl\w*_sync|__ballot_sync|cuda::atomic', wzp)
(out / 'wzp_hip.cuh').write_text('#pragma once\n' + wzp + '\n} } // namespaces\n')
