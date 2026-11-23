#include "small_pwe.cuh"
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <limits>
#include <algorithm>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>T* gpu(size_t n){T* p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>void test(unsigned n,double scale){
    const double eps=0.1,step=eps/scale;
    std::vector<T> orig(n),base(n),err(n);std::vector<unsigned> idx(n);
    for(unsigned i=0;i<n;++i){
        // Mix ordinary residuals, code overflow, large offsets and tiny native
        // ulps. Alternate sign; group boundaries and partial groups are varied.
        orig[i]=i%7==0?T(1e12):i%7==1?T(-100):i%7==2?T(1e-20):T(1);
        double delta=(i%5==0?100.0:i%5==1?0.10001:i%5==2?0.2:i%5==3?0.3:0.6)*(i%2?-1:1);
        base[i]=T(orig[i]+T(delta));err[i]=T(base[i]-orig[i]);idx[i]=i;
    }
    auto* o=gpu<T>(n);auto* b=gpu<T>(n);auto* e=gpu<T>(n);auto* ix=gpu<unsigned>(n);
    unsigned groups=(n+31)/32;size_t prefix=24+groups*8;
    auto* archive=gpu<unsigned char>(prefix+n*sizeof(T)+16);auto* count=gpu<unsigned>(1);
    ck(cudaMemcpy(o,orig.data(),n*sizeof(T),cudaMemcpyHostToDevice));ck(cudaMemcpy(b,base.data(),n*sizeof(T),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(e,err.data(),n*sizeof(T),cudaMemcpyHostToDevice));ck(cudaMemcpy(ix,idx.data(),n*4,cudaMemcpyHostToDevice));ck(cudaMemset(count,0,4));
    // The asynchronous pack must clamp a device count exceeding capacity too.
    const unsigned recorded=n+31U;
    auto* device_n=gpu<unsigned>(1);ck(cudaMemcpy(device_n,&recorded,4,cudaMemcpyHostToDevice));
    WALTZ::pack_small_pwe_async<T><<<2,256>>>(o,ix,e,device_n,n,step,archive,count);
    unsigned payload=0;ck(cudaMemcpy(&payload,count,4,cudaMemcpyDeviceToHost));
    WALTZ::SmallPweHeader h;h.groups=groups;h.payload_bytes=payload;h.step=step;ck(cudaMemcpy(archive,&h,24,cudaMemcpyHostToDevice));
    WALTZ::apply_small_pwe<T><<<(n+255)/256,256>>>(b,ix,archive,n,prefix+payload);ck(cudaDeviceSynchronize());
    std::vector<T> out(n);std::vector<WALTZ::SmallPweGroup> desc(groups);
    ck(cudaMemcpy(out.data(),b,n*sizeof(T),cudaMemcpyDeviceToHost));ck(cudaMemcpy(desc.data(),archive+24,groups*8,cudaMemcpyDeviceToHost));
    unsigned small=0,raw=0;
    for(unsigned i=0;i<n;++i){auto d=desc[i/32];bool native=(d.offset>>31)||((d.native_mask>>(i%32))&1U);
        if(native){++raw;T expected=T(base[i]-err[i]);if(std::memcmp(&out[i],&expected,sizeof(T)))throw std::runtime_error("native mismatch");}
        else{++small;if(fabsl((long double)out[i]-orig[i])>(long double)step)throw std::runtime_error("small repair exceeds safety budget");}
    }
    std::cout<<"PASS bytes="<<sizeof(T)<<" n="<<n<<" scale="<<scale<<" small="<<small<<" raw="<<raw<<"\n";
    ck(cudaFree(o));ck(cudaFree(b));ck(cudaFree(e));ck(cudaFree(ix));ck(cudaFree(archive));ck(cudaFree(count));
    ck(cudaFree(device_n));
}
int main(){try{for(unsigned n:{1,3,7,31,32,33,255,256,257,1025,131073})for(double scale:{16,64}){test<float>(n,scale);test<double>(n,scale);}}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
