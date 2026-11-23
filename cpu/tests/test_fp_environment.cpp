#include "waltz_cpu.h"
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

template<class T> bool check_environment() {
    constexpr unsigned n=17*19*23;
    waltz_cpu_config c{17,19,23,sizeof(T)==4?WALTZ_CPU_F32:WALTZ_CPU_F64,0.001,WALTZ_CPU_SCALAR};
    const size_t capacity=waltz_cpu_max_compressed_bytes(&c);
    T* input=static_cast<T*>(std::malloc(n*sizeof(T)));
    T* first=static_cast<T*>(std::malloc(n*sizeof(T)));
    T* second=static_cast<T*>(std::malloc(n*sizeof(T)));
    void* a=std::malloc(capacity);void* b=std::malloc(capacity);
    auto* w=waltz_cpu_create();bool good=input&&first&&second&&a&&b&&w;
    if(good)for(unsigned i=0;i<n;++i)input[i]=T(int(i%97)-48)/T(16);
    for(int backend=1;good&&backend<=2;++backend) {
        if(backend==2&&!waltz_cpu_has_avx2())continue;
        c.backend=static_cast<waltz_cpu_backend>(backend);size_t as=0,bs=0;
        good=!waltz_cpu_compress(w,&c,input,a,capacity,&as)&&!waltz_cpu_decompress(w,&c,a,as,first,n*sizeof(T));
        const int modes[]={FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO};
        for(int rounding:modes) {
            if(!good)break;
            fenv_t saved;std::fegetenv(&saved);std::fesetround(rounding);
#if defined(__x86_64__) || defined(__i386__)
            const unsigned original=_mm_getcsr();_mm_setcsr(original|(1u<<15)|(1u<<6));
            const unsigned modified=_mm_getcsr();
#endif
            good=!waltz_cpu_compress(w,&c,input,b,capacity,&bs)&&!waltz_cpu_decompress(w,&c,b,bs,second,n*sizeof(T));
            good=good&&std::fegetround()==rounding;
#if defined(__x86_64__) || defined(__i386__)
            good=good&&_mm_getcsr()==modified;_mm_setcsr(original);
#endif
            std::fesetenv(&saved);
            good=good&&as==bs&&!std::memcmp(a,b,as)&&!std::memcmp(first,second,n*sizeof(T));
        }
    }
    if(!good)std::fprintf(stderr,"FP environment failure f%zu: %s\n",sizeof(T)*8,waltz_cpu_last_error(w));
    waltz_cpu_destroy(w);std::free(input);std::free(first);std::free(second);std::free(a);std::free(b);return good;
}
int main(){const bool pass=check_environment<float>()&&check_environment<double>();std::puts(pass?"FP rounding/FTZ/DAZ isolation and restoration PASS":"FAIL");return pass?0:1;}
