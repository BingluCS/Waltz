#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include "transformers/all_halo.cuh"
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Emit {
 T* p;
 __device__ void operator()(T v,size_t i,int,int,int)const{p[i]=v;}
};
static int levels(int n){int l=0;while(n>8 && l<5){n=(n+1)/2;++l;}return l;}
template<class T>void line(std::vector<T>& v){
 const int ne=(v.size()+1)/2,no=v.size()/2;if(!no)return;
 std::vector<T> e(ne),o(no);
 for(int i=0;i<ne;++i)e[i]=v[2*i];for(int i=0;i<no;++i)o[i]=v[2*i+1];
 for(int i=0;i<no;++i)o[i]=std::fma(warp97::Constants<T>::A,T(e[i]+e[std::min(i+1,ne-1)]),o[i]);
 for(int i=0;i<ne;++i)e[i]=std::fma(warp97::Constants<T>::B,T(o[std::max(i-1,0)]+o[std::min(i,no-1)]),e[i]);
 for(int i=0;i<no;++i)o[i]=std::fma(warp97::Constants<T>::G,T(e[i]+e[std::min(i+1,ne-1)]),o[i]);
 for(int i=0;i<ne;++i)e[i]=std::fma(warp97::Constants<T>::D,T(o[std::max(i-1,0)]+o[std::min(i,no-1)]),e[i]);
 for(int i=0;i<ne;++i)v[i]=e[i]*warp97::Constants<T>::K;
 for(int i=0;i<no;++i)v[ne+i]=o[i]*(-warp97::Constants<T>::IK);
}
template<class T>void axis(std::vector<T>& data,dim3 full,dim3 n,int a){
 const auto idx=[&](int x,int y,int z){return (size_t(z)*full.y+y)*full.x+x;};
 const int length=a==0?n.x:a==1?n.y:n.z;
 std::vector<T> v(length);
 const int aa=a==0?n.y:n.x,bb=a==2?n.y:n.z;
 for(int j=0;j<bb;++j)for(int i=0;i<aa;++i){
  for(int k=0;k<length;++k)v[k]=data[a==0?idx(k,i,j):a==1?idx(i,k,j):idx(i,j,k)];
  line(v);
  for(int k=0;k<length;++k)data[a==0?idx(k,i,j):a==1?idx(i,k,j):idx(i,j,k)]=v[k];
 }
}
template<class T>void check(dim3 d){
 using namespace WALTZ::allhalo;
 const size_t count=size_t(d.x)*d.y*d.z,bytes=count*sizeof(T);
 std::vector<T> input(count),cpu,gpu(count),back(count);
 for(size_t i=0;i<count;++i)input[i]=T(.5*std::sin(i*.071)+.3*std::cos(i*.013)+(i%37==0?2.0:0.0));
 cpu=input;T *a,*b,*work;ck(cudaMalloc(&a,bytes));ck(cudaMalloc(&b,bytes));ck(cudaMalloc(&work,bytes));
 ck(cudaMemcpy(a,input.data(),bytes,cudaMemcpyHostToDevice));
 const int lx=levels(std::min(d.x,d.y)),lz=levels(d.z);const bool dyadic=lx==lz;
 const auto step=[&](dim3 active,int axes){
  // Untimed reference fixture preserves every untouched high band explicitly.
  ck(cudaMemcpy(b,a,bytes,cudaMemcpyDeviceToDevice));
  const Read<T> read{a,nullptr,d,dim3(0,0,0)};
  if(axes==3){
   constexpr int Z=sizeof(T)==4?8:4;
   forward_xyz<Z><<<dim3((active.x+55)/56,(active.y+7)/8,(active.z+Z-1)/Z),256,56*8*(Z+8)*sizeof(T)>>>(read,active,Emit<T>{b});
   axis(cpu,d,active,1);axis(cpu,d,active,0);axis(cpu,d,active,2);
  }else if(axes==2){
   xy<T,false><<<dim3((active.x+55)/56,(active.y+7)/8,(active.z+3)/4),128>>>(read,active,Emit<T>{b});
   axis(cpu,d,active,1);axis(cpu,d,active,0);
  }else{
   constexpr int Z=sizeof(T)==4?16:8;
   zline<Z,T,false><<<dim3((active.x*active.y+255)/256,(active.z+Z-1)/Z),256>>>(read,active,Emit<T>{b});
   axis(cpu,d,active,2);
  }
  ck(cudaGetLastError());ck(cudaDeviceSynchronize());std::swap(a,b);
 };
 dim3 active=d;
 if(dyadic)for(int l=0;l<lx;++l){step(active,3);active=shrink(active,3);}
 else{
  for(int l=0;l<lz;++l){step(active,1);active=shrink(active,1);}
  active=d;for(int l=0;l<lx;++l){step(active,2);active=shrink(active,2);}
 }
 ck(cudaMemcpy(gpu.data(),a,bytes,cudaMemcpyDeviceToHost));
 double max_coeff=0,scale=0;
 for(size_t i=0;i<count;++i){max_coeff=std::max(max_coeff,std::abs(double(gpu[i])-double(cpu[i])));scale=std::max(scale,std::abs(double(cpu[i])));}
 const double tol=sizeof(T)==4?2e-5:2e-13;
 if(max_coeff>tol*(1+scale))throw std::runtime_error("independent lifting coefficient mismatch");
 ck(cudaMemset(b,0xff,bytes));ck(cudaMemset(work,0xff,bytes));
 inverse(a,work,b,d,lx,lz,dyadic,Store<T>{b},0);ck(cudaDeviceSynchronize());
 ck(cudaMemcpy(back.data(),b,bytes,cudaMemcpyDeviceToHost));double worst=0;
 for(size_t i=0;i<count;++i){if(!std::isfinite(back[i]))throw std::runtime_error("nonfinite inverse");worst=std::max(worst,std::abs(double(back[i])-double(input[i])));}
 if(worst>(sizeof(T)==4?3e-5:3e-13))throw std::runtime_error("unquantized roundtrip mismatch");
 std::cout<<"PASS bytes="<<sizeof(T)<<" shape="<<d.x<<","<<d.y<<","<<d.z<<" levels="<<lx<<","<<lz
  <<" coeff_error="<<max_coeff<<" roundtrip_error="<<worst<<"\n";
 ck(cudaFree(a));ck(cudaFree(b));ck(cudaFree(work));
}
namespace inverse_address {
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
} // namespace inverse_address

int main()try{
 for(int n=64;n<=4096;++n)for(int p=-4;p<n+59;++p)
  if(halo3d::coeff(p,n)!=halo3d::inverse_coeff<true>(p,n))throw std::runtime_error("address mismatch");
 for(dim3 d:{dim3(64,64,64),dim3(65,67,69),dim3(112,64,64),dim3(127,129,65),dim3(257,128,99)}){
  inverse_address::check<float>(d);inverse_address::check<double>(d);
 }
 for(dim3 d:{dim3(12,12,12),dim3(24,24,24),dim3(48,48,48),dim3(96,96,96),dim3(192,192,192),
   dim3(9,9,9),dim3(16,16,16),dim3(65,65,65),dim3(65,67,33),dim3(128,128,9),
   dim3(17,19,129),dim3(1,1,32),dim3(32,32,1),dim3(7,7,7),dim3(1,17,65),
   dim3(33,35,98),dim3(33,35,100),dim3(2,2,32),dim3(17,19,8193),dim3(32,32,3)}) {
   check<float>(d);check<double>(d);
 }
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
