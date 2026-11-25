// Optional host-side regression. Link the HIP device library and the Waltz CPU
// library; the test itself does not need a HIP device-code compiler.
#include "waltz_hip.h"
#include "waltz_cpu.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
static void ok(bool cond,const char* message){if(!cond)throw std::runtime_error(message);}
static void ck(hipError_t e){ok(e==hipSuccess,hipGetErrorString(e));}
struct Dev {void* data=nullptr;explicit Dev(size_t bytes){ck(hipMalloc(&data,bytes));}~Dev(){(void)hipFree(data);}};
struct Shape {unsigned x,y,z;};
template<class T> void test(waltz_hip_session* gpu,waltz_cpu_workspace* cpu,Shape d,unsigned pattern,unsigned shift=0){
    const size_t n=size_t(d.x)*d.y*d.z,raw=n*sizeof(T);
    const double eb=.001;
    std::vector<T> src(n),reference(n),got(n);
    for(unsigned i=0;i<n;++i){
        // Use dyadic rationals so host math does not vary among toolchains.
        unsigned h=i*747796405u+2891336453u; h=((h>>((h>>28)+4u))^h)*277803737u;
        src[i]=pattern==0?T(0):T(int((h^(h>>22))&1023u)-512)*T(0.03125);
        // Deliberately generate nonempty/outlying coefficients on boundary bits.
        if(pattern && (i%2048==0 || i%2048==31 || i%2048==32 || i%2048==2047))src[i]+=T(700);
        // Mixed sign populations, from empty to dense, with unaligned byte
        // starts. The 1-row shapes have no transform: magnitude entropy here
        // is not smoothed by synthesis. All original values are dyadic.
        if(pattern==2){
            unsigned g=i/16, lane=i%16;
            unsigned pop=(g^(g>>4))%17;
            src[i]=lane<pop ? T((h&1)?1:-1)*T((h>>1)%65535+1)*T(1.0/1024) : T(0);
        }
        if(pattern==3)src[i]=T((h&1)?1:-1)*T((h>>1)%65535+1)*T(1.0/1024);
    }
    waltz_cpu_config c{d.x,d.y,d.z,sizeof(T)==4?WALTZ_CPU_F32:WALTZ_CPU_F64,eb,WALTZ_CPU_AUTO};
    waltz_hip_config g{d.x,d.y,d.z,int(sizeof(T)),eb};
    size_t rcpu=0,rgpu=0; const size_t cap=waltz_hip_max_bytes(&g);
    std::vector<unsigned char> wire(waltz_cpu_max_compressed_bytes(&c));
    ok(waltz_cpu_compress(cpu,&c,src.data(),wire.data(),wire.size(),&rcpu)==0,waltz_cpu_last_error(cpu));
    ok(waltz_cpu_decompress(cpu,&c,wire.data(),rcpu,reference.data(),raw)==0,waltz_cpu_last_error(cpu));
    Dev input(raw+shift),output(raw+shift),archive(cap);
    auto* in=static_cast<unsigned char*>(input.data)+shift;
    auto* out=static_cast<unsigned char*>(output.data)+shift;
    ck(hipMemcpy(in,src.data(),raw,hipMemcpyHostToDevice));
    for(unsigned repeat=0;repeat<2;++repeat){
        ck(hipMemset(archive.data,0xa5,cap));
        ok(waltz_hip_encode(gpu,&g,in,archive.data,cap,&rgpu)==0,waltz_hip_error(gpu));
        ok(rgpu==rcpu,"archive size differs");
        std::vector<unsigned char> other(rgpu);
        ck(hipMemcpy(other.data(),archive.data,rgpu,hipMemcpyDeviceToHost));
        ok(!std::memcmp(other.data(),wire.data(),rgpu),"archive bytes differ");
    }
    // Feed the independently encoded CPU stream to the DCU decoder.
    ck(hipMemcpy(archive.data,wire.data(),rcpu,hipMemcpyHostToDevice));
    ok(waltz_hip_decode(gpu,&g,archive.data,rcpu,out,raw)==0,waltz_hip_error(gpu));
    ck(hipMemcpy(got.data(),out,raw,hipMemcpyDeviceToHost));
    ok(!std::memcmp(reference.data(),got.data(),raw),"decoded bits differ");
    for(size_t i=0;i<n;++i)ok(std::isfinite(got[i])&&std::fabs((long double)got[i]-src[i])<=eb,"bound failed");
}
int main() try{
    ck(hipSetDevice(0));
    auto gpu=std::unique_ptr<waltz_hip_session,decltype(&waltz_hip_destroy)>(waltz_hip_create(0),waltz_hip_destroy);
    auto cpu=std::unique_ptr<waltz_cpu_workspace,decltype(&waltz_cpu_destroy)>(waltz_cpu_create(),waltz_cpu_destroy);
    ok(gpu&&cpu,"session creation");unsigned tests=0;
    for(Shape d:{Shape{1,1,1},{31,1,1},{32,1,1},{33,1,1},{23,89,1},{16,16,8},{3,683,1},
                 {95,83,2},{139,101,3},{139,101,4},
                 {127,63,16},{128,64,16},{129,65,17},{128,128,64},
                 {128,128,119},{132,129,135},{136,129,143},{130,128,121}}){
        for(unsigned p:{1u,0u}){
            std::printf("testing %ux%ux%u pattern %u\n",d.x,d.y,d.z,p);
            test<float>(gpu.get(),cpu.get(),d,p);test<double>(gpu.get(),cpu.get(),d,p);tests+=2;
        }
    }
    for(Shape d:{Shape{8191,1,1},{8192,1,1},{8193,1,1},{255,131,1},{129,129,25}}){
        for(unsigned p:{2u,3u}){
            std::printf("testing %ux%ux%u pattern %u\n",d.x,d.y,d.z,p);
            test<float>(gpu.get(),cpu.get(),d,p);test<double>(gpu.get(),cpu.get(),d,p);tests+=2;
        }
    }
    // Native FP32 pointers need not be float2 aligned.
    for(unsigned p:{0u,1u}){test<float>(gpu.get(),cpu.get(),{128,128,119},p,sizeof(float));++tests;}
    std::printf("PASS %u cases\n",tests);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
