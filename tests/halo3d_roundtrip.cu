#include "waltz.hpp"
#include "utils/Blob.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class T> static void check_case(dim3 dims,const char* mode) {
    const size_t n=size_t(dims.x)*dims.y*dims.z,bytes=n*sizeof(T);
    std::vector<T> input(n),out(n),first(n);
    double lo=INFINITY,hi=-INFINITY;
    for(size_t j=0;j<n;++j){input[j]=T(100+3*std::sin(j*.071)+.7*std::cos(j*.019)+(j%113==0?10:0));lo=std::min(lo,double(input[j]));hi=std::max(hi,double(input[j]));}
    const double eps=(hi-lo)*1e-3;
    WALTZ::Config cfg(dims.x,dims.y,dims.z);cfg.errorBoundMode=WALTZ::EB_ABS;cfg.absErrorBound=eps;
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    T *src=nullptr,*dst=nullptr;char* archive=nullptr;
    ck(cudaMalloc(&src,bytes));ck(cudaMalloc(&dst,bytes));ck(cudaMalloc(&archive,WALTZ::compressed_buffer_capacity<T>(n)));
    ck(cudaMemcpyAsync(src,input.data(),bytes,cudaMemcpyHostToDevice,stream));
    cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));
    for(int repeat=0;repeat<2;++repeat) {
        setenv("WALTZ_HALO3D",mode,1);
        const size_t packed=WALTZ::d_compress<T>(cfg,src,archive,nullptr,stream);
        WALTZ::WaltzBlobHeader h{};ck(cudaMemcpyAsync(&h,archive,sizeof(h),cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
        const bool all=std::string(mode)=="both" || std::string(mode)=="auto" || std::string(mode)=="all";
        const bool requested=std::string(mode)=="inverse";
        const bool expected=requested && h.use_dyadic() && h.levels_xy && prop.sharedMemPerBlockOptin>=65536U;
        const auto synthesis=all?WALTZ::Synthesis::HaloAllLevels:
            expected?WALTZ::Synthesis::HaloLifting3D:WALTZ::Synthesis::Convolution;
        require(h.synthesis()==synthesis,"wrong archived synthesis");
        // The decoder must not consult the compression environment. Poison the
        // destination and use a distinct device archive, not encoder leftovers.
        std::vector<char> saved(packed);ck(cudaMemcpyAsync(saved.data(),archive,packed,cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
        char* fresh=nullptr;ck(cudaMalloc(&fresh,packed));ck(cudaMemcpyAsync(fresh,saved.data(),packed,cudaMemcpyHostToDevice,stream));
        ck(cudaMemsetAsync(dst,0xff,bytes,stream));setenv("WALTZ_HALO3D","ignored-on-decode",1);
        WALTZ::d_decompress<T>(cfg,fresh,dst,nullptr,stream);
        ck(cudaMemcpyAsync(out.data(),dst,bytes,cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
        for(size_t j=0;j<n;++j) require(std::isfinite(out[j]) && fabsl(static_cast<long double>(out[j])-input[j])<=static_cast<long double>(eps),"strict pointwise failure");
        if(repeat)require(std::memcmp(first.data(),out.data(),bytes)==0,"repeated compression changed reconstruction");else first=out;
        for (uint8_t bad_flags : {uint8_t(0),uint8_t(0x20),uint8_t(0x57),uint8_t(255)}) {
            auto invalid=h;invalid.flags=bad_flags;
            ck(cudaMemcpyAsync(fresh,&invalid,sizeof(invalid),cudaMemcpyHostToDevice,stream));
            bool rejected=false;try{WALTZ::d_decompress<T>(cfg,fresh,dst,nullptr,stream);}catch(const std::runtime_error&){rejected=true;}
            require(rejected,"invalid archive flags accepted");
        }
        ck(cudaFree(fresh));
    }
    setenv("WALTZ_HALO3D","invalid",1);bool rejected=false;
    try{WALTZ::d_compress<T>(cfg,src,archive,nullptr,stream);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"unknown compression mode accepted");
    unsetenv("WALTZ_HALO3D");
    ck(cudaFree(src));ck(cudaFree(dst));ck(cudaFree(archive));ck(cudaStreamDestroy(stream));
    std::cout<<"PASS halo type="<<sizeof(T)<<" shape="<<dims.x<<"x"<<dims.y<<"x"<<dims.z<<" mode="<<mode<<"\n";
}
int main() try {
    ck(cudaSetDevice(0));WALTZ::Waltz_report_enabled=false;
    for(const char* mode:{"0","auto","all","forward","inverse","both"}) {
        check_case<float>(dim3(65,65,65),mode);
        check_case<float>(dim3(65,67,33),mode);
        check_case<double>(dim3(64,64,64),mode);
        check_case<double>(dim3(65,67,33),mode);
    }
    check_case<float>(dim3(257,257,257),"auto");
    check_case<float>(dim3(128,128,128),"auto");
    check_case<double>(dim3(128,128,128),"auto");
    check_case<double>(dim3(256,256,256),"auto");
    // Large forward-only must not consume flags produced only by convolution.
    check_case<double>(dim3(256,256,256),"forward");
    // Full Morton bricks with non-power-of-two BZ must not use bit-shift
    // addressing. 192^3/48^3 choose BZ=6; plane 32x32x3 chooses BZ=3.
    for(dim3 d:{dim3(192,192,192),dim3(48,48,48),dim3(32,32,3),dim3(32,32,12),dim3(17,19,8193)}) {
        check_case<float>(d,"auto");check_case<double>(d,"auto");
    }
    for(dim3 d:{dim3(9,9,9),dim3(7,7,7),dim3(2,2,32),dim3(32,32,2),dim3(17,19,129)}) {
        check_case<float>(d,"all");check_case<double>(d,"all");
    }
    for(dim3 d:{dim3(32,32,98),dim3(32,32,100)}) {
        check_case<float>(d,"all");check_case<double>(d,"all");
    }
    unsetenv("WALTZ_HALO3D");
} catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
