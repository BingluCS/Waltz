#include <cuda_runtime.h>
#include <cstddef>
#include "transformers/halo3d.cuh"
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstring>
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Store {
 T* p;
 __device__ void operator()(T v,size_t i)const{p[i]=v;}
};
template<class T> void check(dim3 d){
 constexpr int Z=sizeof(T)==4?16:8;
 const size_t n=size_t(d.x)*d.y*d.z, bytes=n*sizeof(T);
 std::vector<T> src(n),a(n),b(n);
 for(size_t i=0;i<n;++i)src[i]=T(int((i*137)%1237)-618)/T(32);
 T *s,*x,*y;ck(cudaMalloc(&s,bytes));ck(cudaMalloc(&x,bytes));ck(cudaMalloc(&y,bytes));
 ck(cudaMemcpy(s,src.data(),bytes,cudaMemcpyHostToDevice));
 ck(cudaMemset(x,0x55,bytes));ck(cudaMemset(y,0xaa,bytes));
 ck(cudaFuncSetAttribute(halo3d::inverse<Z,T,Store<T>,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536));
 ck(cudaFuncSetAttribute(halo3d::inverse<Z,T,Store<T>,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536));
 dim3 grid((d.x+55)/56,(d.y+7)/8,(d.z+Z-1)/Z);
 halo3d::inverse<Z,T,Store<T>,false><<<grid,256,65536>>>(s,d.x,d.y,d.z,Store<T>{x});
 halo3d::inverse<Z,T,Store<T>,true><<<grid,256,65536>>>(s,d.x,d.y,d.z,Store<T>{y});
 ck(cudaGetLastError());ck(cudaDeviceSynchronize());
 ck(cudaMemcpy(a.data(),x,bytes,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),y,bytes,cudaMemcpyDeviceToHost));
 if(std::memcmp(a.data(),b.data(),bytes))throw std::runtime_error("inverse bitwise mismatch");
 ck(cudaFree(s));ck(cudaFree(x));ck(cudaFree(y));
 std::cout<<"PASS inverse bitwise bytes="<<sizeof(T)<<" shape="<<d.x<<","<<d.y<<","<<d.z<<"\n";
}
int main()try{
 for(int n=64;n<=4096;++n)for(int p=-4;p<n+59;++p)
  if(halo3d::coeff(p,n)!=halo3d::inverse_coeff<true>(p,n))throw std::runtime_error("address mismatch");
 for(dim3 d:{dim3(64,64,64),dim3(65,67,69),dim3(112,64,64),dim3(127,129,65),dim3(257,128,99)}){
  check<float>(d);check<double>(d);
 }
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
