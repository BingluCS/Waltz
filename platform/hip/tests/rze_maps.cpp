// Standalone gfx936 property test for the large-block decoder; build with hipcc.
#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <stdexcept>
#include "wzp_hip.cuh"

static void ck(hipError_t e) { if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e)); }
__global__ __launch_bounds__(TPB) void restore_maps(const byte* all,const unsigned* lengths,byte* result) {
    using namespace WALTZ::lossless;
    __shared__ __align__(16) byte scratch[TEMP_BYTES], restore[CS];
    const unsigned k=blockIdx.x;
    rze_full(all+k*CS,lengths[k],restore,scratch);
    __syncthreads();
    for(unsigned j=threadIdx.x;j<CS;j+=TPB)result[k*CS+j]=restore[j];
}
// Deliberately independent byte-at-a-time encoder of the LC/WZP wire
// contract: leaf bits mean nonzero; at the next three levels bits mean
// changed since the preceding value (initial value zero at each level).
static std::vector<byte> pack_map(const byte* raw) {
    std::vector<byte> bitmap(CS/8),wire;
    for(unsigned j=0;j<CS;++j)if(raw[j]){
        bitmap[j/8]|=byte(1u<<(j%8));wire.push_back(raw[j]);
    }
    while(bitmap.size()>4){
        std::vector<byte> upper(bitmap.size()/8);byte prev=0;
        for(size_t j=0;j<bitmap.size();++j){
            if(prev!=bitmap[j]){upper[j/8]|=byte(1u<<(j%8));wire.push_back(bitmap[j]);}
            prev=bitmap[j];
        }
        bitmap.swap(upper);
    }
    wire.insert(wire.end(),bitmap.begin(),bitmap.end());return wire;
}
int main() try {
    constexpr unsigned N=320;
    std::vector<byte> raw(N*CS),packed(N*CS),out(N*CS);
    std::vector<unsigned> len(N);
    for(unsigned k=0;k<N;++k){
        unsigned s=k+1;
        for(unsigned j=0;j<CS;++j){
            s^=s<<13;s^=s>>17;s^=s<<5;
            bool present=(s%256)<(k%193);
            if(k<64)present=j==((k*509u)%CS);
            if(k>=64&&k<96)present=(j%8==k%8 && j/128>=k-64);
            if(k>=96&&k<128)present=(j/32%(k%15+2)==0 && (j/1024)%2);
            if(k==128)present=(j==CS-1);
            raw[k*CS+j]=present?byte(1+(s>>16)%255):0;
        }
        auto p=pack_map(raw.data()+k*CS);
        if(p.size()>=CS)throw std::runtime_error("test stream too large");
        len[k]=p.size();std::memcpy(packed.data()+k*CS,p.data(),p.size());
    }
    byte *din=nullptr,*dout=nullptr;unsigned* dlen=nullptr;
    ck(hipMalloc(&din,raw.size()));ck(hipMalloc(&dout,raw.size()));ck(hipMalloc(&dlen,N*sizeof(unsigned)));
    ck(hipMemcpy(din,packed.data(),raw.size(),hipMemcpyHostToDevice));
    ck(hipMemcpy(dlen,len.data(),N*sizeof(unsigned),hipMemcpyHostToDevice));
    restore_maps<<<N,TPB>>>(din,dlen,dout);ck(hipGetLastError());
    ck(hipMemcpy(out.data(),dout,raw.size(),hipMemcpyDeviceToHost));
    ck(hipFree(din));ck(hipFree(dout));ck(hipFree(dlen));
    if(out!=raw)throw std::runtime_error("expanded bytes differ");
    std::printf("PASS %u independent full-block bitmap streams\n",N);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
