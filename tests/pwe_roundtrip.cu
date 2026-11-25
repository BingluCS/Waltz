#include "waltz.hpp"
#include "utils/Blob.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class T> T* gpu(size_t n) {
    T* data = nullptr;
    ck(cudaMalloc(&data, n*sizeof(T)));
    return data;
}
static size_t align8(size_t n) { return (n+7U)&~size_t(7U); }
static unsigned compact_cases = 0, native_cases = 0, zero_cases = 0;

template<class T> void roundtrip(dim3 dims, bool constant, double offset, bool short_plane_z=false) {
    const size_t n = size_t(dims.x)*dims.y*dims.z;
    std::vector<T> input(n), output(n), again(n);
    double low = INFINITY, high = -INFINITY;
    for (size_t i=0; i<n; ++i) {
        input[i] = constant ? T(100) : T(offset + std::sin(i*0.071)*3.0 +
            std::cos(i*0.019)*0.7 + (i%113==0 ? 10.0 : 0.0));
        low = std::min(low,double(input[i])); high = std::max(high,double(input[i]));
    }
    const double bound = constant ? 0.01 : (high-low)*1e-3;
    WALTZ::Config config(dims.x,dims.y,dims.z);
    config.errorBoundMode = WALTZ::EB_ABS;
    config.absErrorBound = bound;
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    T* source=gpu<T>(n), *decoded=gpu<T>(n);
    const size_t capacity=WALTZ::compressed_buffer_capacity<T>(n);
    char* archive=gpu<char>(capacity);
    ck(cudaMemcpyAsync(source,input.data(),n*sizeof(T),cudaMemcpyHostToDevice,stream));
    for (const char* mode : {"0","16","64","default"}) {
        if (std::string(mode)=="default") unsetenv("WALTZ_SMALL_PWE_SCALE");
        else setenv("WALTZ_SMALL_PWE_SCALE",mode,1);
        const double scale=std::string(mode)=="default" ? 0.0 : std::stod(mode);
        const size_t bytes=WALTZ::d_compress<T>(config,source,archive,nullptr,stream);
        ck(cudaStreamSynchronize(stream));
        require(bytes<=capacity,"compressed capacity exceeded");
        WALTZ::WaltzBlobHeader header{};
        ck(cudaMemcpy(&header,archive,sizeof(header),cudaMemcpyDeviceToHost));
        const auto z_flag=WALTZ::WaltzBlobHeader::CudaFusedPlaneZ;
        require(WALTZ::WaltzBlobHeader::valid_flags(header.flags&~z_flag) &&
                header.element_bytes()==sizeof(T),"wrong Waltz header");
        require(bool(header.flags&z_flag)==(short_plane_z&&scale>0),
                "wrong default or portable plane-Z dispatch");
        if (header.pwe_encoding() == WALTZ::PweEncoding::Small) {
            ++compact_cases;
            require(scale>0,"disabled mode generated WPW1");
            require(header.pwe_bytes<size_t(header.n_pwe)*sizeof(T),"compact representation expanded");
            const size_t error_offset=align8(align8(sizeof(header)+header.mag_comp_bytes)+size_t(header.n_pwe)*4U);
            WALTZ::SmallPweHeader small{};
            ck(cudaMemcpy(&small,archive+error_offset,sizeof(small),cudaMemcpyDeviceToHost));
            require(small.magic==WALTZ::SMALL_PWE_MAGIC && small.step==bound/scale,"wrong stored precision");
        } else {
            ++native_cases;
            require(header.pwe_bytes==size_t(header.n_pwe)*sizeof(T),"wrong native residual length");
        }
        if (!header.n_pwe) ++zero_cases;
        // Decoder uses the archived precision, never this compression setting.
        setenv("WALTZ_SMALL_PWE_SCALE","not-a-compression-setting",1);
        WALTZ::d_decompress<T>(config,archive,decoded,nullptr,stream);
        ck(cudaMemcpy(output.data(),decoded,n*sizeof(T),cudaMemcpyDeviceToHost));
        for (size_t i=0; i<n; ++i) {
            long double error=fabsl(static_cast<long double>(output[i])-input[i]);
            require(std::isfinite(output[i]) && error<=static_cast<long double>(bound),"strict pointwise failure");
        }
        std::vector<char> saved(bytes);
        ck(cudaMemcpy(saved.data(),archive,bytes,cudaMemcpyDeviceToHost));
        size_t section_offset=align8(sizeof(header));
        const size_t lengths[]={size_t(header.mag_comp_bytes),size_t(header.n_pwe)*4U,
            size_t(header.pwe_bytes),size_t(header.n_qo)*4U,size_t(header.n_qo)*4U};
        for(size_t length:lengths) {
            const size_t end=section_offset+length;
            for(size_t k=end;k<align8(end);++k)require(saved[k]==0,"uninitialized section padding");
            section_offset=align8(end);
        }
        require(section_offset==bytes,"archive section size mismatch");
        ck(cudaMemsetAsync(archive,0,capacity,stream));
        ck(cudaMemsetAsync(decoded,0xff,n*sizeof(T),stream));
        ck(cudaMemcpyAsync(archive,saved.data(),bytes,cudaMemcpyHostToDevice,stream));
        WALTZ::d_decompress<T>(config,archive,decoded,nullptr,stream);
        ck(cudaMemcpy(again.data(),decoded,n*sizeof(T),cudaMemcpyDeviceToHost));
        require(std::memcmp(again.data(),output.data(),n*sizeof(T))==0,"archive round-trip mismatch");
        // Reject invalid flags, not a legacy header identifier.
        auto invalid=header;
        invalid.flags=0;
        ck(cudaMemcpyAsync(archive,&invalid,sizeof(invalid),cudaMemcpyHostToDevice,stream));
        bool rejected=false;
        try { WALTZ::d_decompress<T>(config,archive,decoded,nullptr,stream); }
        catch(const std::runtime_error&) { rejected=true; }
        require(rejected,"invalid archive flags accepted");
        std::cout<<"PASS type="<<sizeof(T)<<" shape="<<dims.x<<"x"<<dims.y<<"x"<<dims.z
                 <<" constant="<<constant<<" offset="<<offset<<" mode="<<mode<<" pwe="<<header.n_pwe<<"\n";
    }
    // Invalid precision is rejected before any GPU work is submitted.
    for (const char* mode : {"1","-1","nan","inf","64x",""}) {
        setenv("WALTZ_SMALL_PWE_SCALE",mode,1);
        bool rejected=false;
        try { WALTZ::d_compress<T>(config,source,archive,nullptr,stream); }
        catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"invalid compression precision was accepted");
    }
    unsetenv("WALTZ_SMALL_PWE_SCALE");
    if (constant) {
        const T nan=std::numeric_limits<T>::quiet_NaN();
        ck(cudaMemcpyAsync(source,&nan,sizeof(nan),cudaMemcpyHostToDevice,stream));
        bool rejected=false;
        try { WALTZ::d_compress<T>(config,source,archive,nullptr,stream); }
        catch(const std::runtime_error&) { rejected=true; }
        require(rejected,"default encoder accepted a nonfinite input");
        rejected=false;
        try { WALTZ::d_compress<T>(config,source,reinterpret_cast<char*>(source),nullptr,stream); }
        catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"default encoder accepted overlapping buffers");
    }
    ck(cudaFree(source));ck(cudaFree(decoded));ck(cudaFree(archive));ck(cudaStreamDestroy(stream));
}

int main() try {
    ck(cudaSetDevice(0));WALTZ::Waltz_report_enabled=false;
    unsetenv("WALTZ_HALO3D");
    for (dim3 dims : {dim3(65,67,33),dim3(64,64,64)}) {
        roundtrip<float>(dims,false,100.0);roundtrip<double>(dims,false,100.0);
        roundtrip<float>(dims,true,0.0);roundtrip<double>(dims,true,0.0);
    }
    roundtrip<float>(dim3(65,67,33),false,1e5);
    roundtrip<double>(dim3(65,67,33),false,1e12);
    const auto& prop=WALTZ::gpu_config().properties;
    const bool short_z_supported=prop.sharedMemPerBlockOptin>=65536U &&
        prop.sharedMemPerMultiprocessor>=3U*65536U;
    // Eligible FP32 plane shape: opt-in compact uses fused Z, default/scale 0
    // keeps bit-portable lifting. The same FP64 geometry keeps lifting.
    roundtrip<float>(dim3(72,70,25),false,100.0,short_z_supported);
    roundtrip<double>(dim3(72,70,25),false,100.0);
    require(compact_cases && native_cases && zero_cases,"missing compact/native/zero coverage");
    std::cout<<"PASS compact="<<compact_cases<<" native="<<native_cases<<" zero="<<zero_cases<<"\n";
} catch (const std::exception& error) {
    std::cerr<<error.what()<<"\n";return 1;
}
