// Compare the two HIP WZP producer inputs at the slot boundary. Tails, zero
// chunks, compressed and RAW chunks must yield identical metadata and bytes.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "wzp_hip.cuh"
namespace {
void ck(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
template<class T> struct Buffer{
    T* p=nullptr;size_t n;
    explicit Buffer(size_t k):n(k){ck(hipMalloc(&p,k*sizeof(T)));}
    ~Buffer(){(void)hipFree(p);}
    void set(const std::vector<T>& v){ck(hipMemcpy(p,v.data(),n*sizeof(T),hipMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);ck(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost));return v;}
};
using Result=std::array<std::vector<byte>,3>;
template<bool Packed> Result run(unsigned n,const std::vector<uint16_t>& input,const std::vector<unsigned>& side){
    const unsigned chunks=(n+8191)/8192;
    Buffer<uint16_t> v(n);v.set(input);
    Buffer<unsigned> s(side.size()),next(1);s.set(side);ck(hipMemset(next.p,0,4));
    Buffer<byte> mag(chunks*CS),sign(chunks*1024),meta(chunks*3);
    ck(hipMemset(mag.p,0xa5,mag.n));ck(hipMemset(sign.p,0xa5,sign.n));
    WALTZ::lossless::encode_kernel<Packed><<<std::min(8u,chunks),TPB>>>(
        reinterpret_cast<byte*>(v.p),n*2,s.p,mag.p,sign.p,meta.p,next.p);
    ck(hipGetLastError());ck(hipDeviceSynchronize());
    return {meta.get(),mag.get(),sign.get()};
}
}
int main() try{
    unsigned cases=0,seen=0;
    for(unsigned n:{1u,31u,32u,511u,8191u,8192u,8193u,16u*8192+333}){
        for(unsigned kind=0;kind<4;++kind){
            std::vector<uint16_t> mag(n),code(n);
            std::vector<unsigned> signs((n+31)/32),escape((n+1)/2,0xdeadbeefu);
            unsigned rng=43;
            for(unsigned i=0;i<n;++i){
                rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;
                const unsigned pattern=(kind+i/8192)%4;
                constexpr unsigned edge[]={0,1,32766,32767,32768,65534,65535,65536,77777};
                unsigned m=pattern==0?0:pattern==1?edge[(i/2)%9]:
                    pattern==2?(rng&65535u):(i%31==0?edge[(i/31)%9]:0);
                m=std::min(65535u,m);
                const bool neg=m && (pattern==2?(rng>>16)&1:i&1);
                mag[i]=m;signs[i/32]|=unsigned(neg)<<(i&31);
                code[i]=std::min(m,32767u)|(unsigned(neg)<<15);
                if(m>=32767){unsigned sh=(i&1)*16;
                    escape[i/2]=(escape[i/2]&~(65535u<<sh))|(m<<sh);}
            }
            auto ref=run<false>(n,mag,signs);
            if(run<true>(n,code,escape)!=ref)throw std::runtime_error("metadata or payload mismatch");
            for(size_t j=0;j<ref[0].size();j+=3){
                unsigned m=(ref[0][j]|(ref[0][j+1]<<8))&0x3fff;
                seen|=m==0?1:m==0x3fff?2:4;
            }
            ++cases;
        }
    }
    if(seen!=7)throw std::runtime_error("missing ZERO/RAW/compressed coverage");
    std::printf("PASS %u paired WZP cases (ZERO, RAW, compressed, escapes and tails)\n",cases);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
