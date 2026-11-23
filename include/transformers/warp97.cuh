#pragma once
#include <cuda_runtime.h>

// Independent readable implementation of the scheduling strategy observed in
// nvJPEG2000 0.11.0.51. NOT NVIDIA source, and NOT a bitwise clone. Native FP32/FP64 I/O,
// Waltz's normalization/sign, full-plane quadrant layout, no level shift/MCT.
// Register and warp-shuffle math shared by the 3D codec kernels.
namespace warp97 {
template<class T> struct Constants {
    static constexpr T A=T(-1.586134342059924), B=T(-0.052980118572961);
    static constexpr T G=T(0.882911075530934), D=T(0.443506852043971);
    static constexpr T K=T(1.1496043988602418), IK=T(1)/K;
};
__device__ __forceinline__ float madd(float a,float b,float c){return __fmaf_rn(a,b,c);}
__device__ __forceinline__ double madd(double a,double b,double c){return __fma_rn(a,b,c);}

template<class T> __device__ __forceinline__ void analysis_y(T (&e)[8], T (&o)[8]) {
    #pragma unroll
    for (int j=0; j<7; ++j) o[j] = madd(Constants<T>::A, e[j]+e[j+1], o[j]);
    #pragma unroll
    for (int j=1; j<7; ++j) e[j] = madd(Constants<T>::B, o[j-1]+o[j], e[j]);
    #pragma unroll
    for (int j=1; j<6; ++j) o[j] = madd(Constants<T>::G, e[j]+e[j+1], o[j]);
    #pragma unroll
    for (int j=2; j<6; ++j) e[j] = madd(Constants<T>::D, o[j-1]+o[j], e[j]);
}

template<class T> __device__ __forceinline__ void analysis_x(T& e, T& o) {
    o = madd(Constants<T>::A, e+__shfl_down_sync(0xffffffff, e, 1), o);
    e = madd(Constants<T>::B, o+__shfl_up_sync(0xffffffff, o, 1), e);
    o = madd(Constants<T>::G, e+__shfl_down_sync(0xffffffff, e, 1), o);
    e = madd(Constants<T>::D, o+__shfl_up_sync(0xffffffff, o, 1), e);
    e *= Constants<T>::K;
    o *= -Constants<T>::IK; // Waltz high-pass sign, unlike JPEG2000's convention.
}

template<class T> __device__ __forceinline__ void synthesis_x(T& e, T& o) {
    e *= Constants<T>::IK; o *= -Constants<T>::K;
    e = madd(-Constants<T>::D, o+__shfl_up_sync(0xffffffff,o,1), e);
    o = madd(-Constants<T>::G, e+__shfl_down_sync(0xffffffff,e,1), o);
    e = madd(-Constants<T>::B, o+__shfl_up_sync(0xffffffff,o,1), e);
    o = madd(-Constants<T>::A, e+__shfl_down_sync(0xffffffff,e,1), o);
}

template<class T> __device__ __forceinline__ void synthesis_y(T (&e)[8], T (&o)[8]) {
    #pragma unroll
    for (int j=0; j<8; ++j) { e[j]*=Constants<T>::IK; o[j]*=-Constants<T>::K; }
    #pragma unroll
    for (int j=1; j<8; ++j) e[j]=madd(-Constants<T>::D,o[j-1]+o[j],e[j]);
    #pragma unroll
    for (int j=1; j<7; ++j) o[j]=madd(-Constants<T>::G,e[j]+e[j+1],o[j]);
    #pragma unroll
    for (int j=2; j<7; ++j) e[j]=madd(-Constants<T>::B,o[j-1]+o[j],e[j]);
    #pragma unroll
    for (int j=2; j<6; ++j) o[j]=madd(-Constants<T>::A,e[j]+e[j+1],o[j]);
}

} // namespace warp97
