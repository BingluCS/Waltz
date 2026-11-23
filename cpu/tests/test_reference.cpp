// Freeze a pre-optimization executable's files, then compare a new executable.
// Does not require a GPU; a successful comparison preserves previously tested
// CPU/GPU compatibility for these inputs, not a new GPU architecture validation.
#define main waltz_codec_acceptance_main
#include "test_codec.cpp"
#undef main

template<class T> bool reference_case(const char* dir,bool write,unsigned nx,unsigned ny,unsigned nz,unsigned pattern,double eb) {
    waltz_cpu_config c{nx,ny,nz,sizeof(T)==4?WALTZ_CPU_F32:WALTZ_CPU_F64,eb,WALTZ_CPU_AVX2};
    const size_t raw=size_t(nx)*ny*nz*sizeof(T),cap=waltz_cpu_max_compressed_bytes(&c);
    Buffer input(raw),output(raw),encoded(cap),saved(cap>raw?cap:raw);Workspace w;
    fill(reinterpret_cast<T*>(input.data),c,pattern);size_t bytes=0;
    if(waltz_cpu_compress(w.value,&c,input.data,encoded.data,cap,&bytes)||
       waltz_cpu_decompress(w.value,&c,encoded.data,bytes,output.data,raw)) {
        std::fprintf(stderr,"codec failed: %s\n",waltz_cpu_last_error(w.value));return false;
    }
    for(unsigned stream=0;stream<2;++stream) {
        char path[4096];
        std::snprintf(path,sizeof(path),"%s/f%zu_%u_%u_%u_p%u_%.0e.%s",dir,sizeof(T)*8,nx,ny,nz,pattern,eb,stream?"raw":"wlt2");
        const size_t size=stream?raw:bytes;const unsigned char* data=stream?output.data:encoded.data;
        FILE* file=std::fopen(path,write?"wbx":"rb");if(!file){std::perror(path);return false;}
        bool good;
        if(write)good=std::fwrite(data,1,size,file)==size;
        else good=std::fread(saved.data,1,size,file)==size&&std::fgetc(file)==EOF&&!std::memcmp(data,saved.data,size);
        if(std::fclose(file))good=false;
        if(!good){std::fprintf(stderr,"reference mismatch: %s\n",path);return false;}
    }
    return true;
}
int main(int argc,char** argv) {
    if(argc!=3||(std::strcmp(argv[1],"write")&&std::strcmp(argv[1],"check"))){std::fprintf(stderr,"Usage: %s write|check existing-directory\n",argv[0]);return 2;}
    const bool write=!std::strcmp(argv[1],"write");unsigned count=0;
    unsigned shapes[][3]={{1,1,1},{17,19,23},{48,32,9},{9,11,65},{64,64,64},{128,96,65}};
    for(auto& s:shapes)for(unsigned p=0;p<4;++p){
        if(!reference_case<float>(argv[2],write,s[0],s[1],s[2],p,1e-3)||!reference_case<double>(argv[2],write,s[0],s[1],s[2],p,1e-3))return 1;
        count+=2;
    }
    if(!reference_case<float>(argv[2],write,17,19,23,2,1e-5)||!reference_case<double>(argv[2],write,17,19,23,2,1e-5))return 1;
    if(!reference_case<float>(argv[2],write,64,64,64,1,1e-5)||!reference_case<double>(argv[2],write,64,64,64,1,1e-5))return 1;
    std::printf("reference %s: %u cases, complete archive and reconstruction bytes PASS\n",write?"written":"checked",count+4);
}
