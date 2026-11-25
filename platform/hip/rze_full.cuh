// Full WZP magnitude chunks carry four final repetition-map bytes, followed
// backwards by change lists for the 32-, 256-, and 2048-byte bitmap levels.
// Inclusive rank selects the most recent changed byte, or zero before the
// first change. The final bitmap means nonzero rather than changed.
template<unsigned W>
__device__ __forceinline__ unsigned rze_scan(unsigned x) {
    #pragma unroll
    for(unsigned d=1;d<W;d*=2){
        unsigned t=__shfl_up(x,d,W);
        if((threadIdx.x&(W-1))>=d)x+=t;
    }
    return x;
}
__device__ __forceinline__ uint2 rze_eight(const byte* data,int pos,unsigned prior,
                                            unsigned change,unsigned limit) {
    uint2 words=make_uint2(0,0);
    #pragma unroll
    for(unsigned i=0;i<8;++i) {
        prior+=(change>>i)&1;
        const unsigned idx=unsigned(pos+int(prior)-1);
        const unsigned v=prior && idx<limit?data[idx]:0;
        if(i<4)words.x|=v<<(8*i);
        else words.y|=v<<(8*(i-4));
    }
    return words;
}
__device__ __forceinline__ void rze_full(const byte* in,unsigned stored,byte* out,byte* temp) {
    static_assert(CS==16384 && TPB==512 && TEMP_BYTES>=2692);
    const unsigned t=threadIdx.x;
    auto* first=reinterpret_cast<unsigned*>(temp+256);  // 2048 bytes
    byte* second=temp+2304;                            // 256 bytes
    auto* before=reinterpret_cast<unsigned*>(temp+2560);// 32 counts and offset
    if(t<32){
        const int top=int(stored)-4;
        unsigned changes=unsigned(in[top])|(unsigned(in[top+1])<<8)|
                         (unsigned(in[top+2])<<16)|(unsigned(in[top+3])<<24);
        const int b3=top-__popc(changes);
        unsigned rank=__popc(changes & (0xffffffffu>>(31-t)));
        const unsigned idx=unsigned(b3+int(rank)-1);
        const unsigned m=rank && idx<stored?in[idx]:0;
        const unsigned count=__popc(m),sum=rze_scan<32>(count);
        const int b2=b3-int(__shfl(sum,31,32));
        const uint2 w=rze_eight(in,b2,sum-count,m,stored);
        reinterpret_cast<uint2*>(second)[t]=w;
        const unsigned count2=__popc(w.x)+__popc(w.y),sum2=rze_scan<32>(count2);
        before[t]=sum2-count2;
        if(t==31)before[32]=unsigned(b2-int(sum2));
    }
    __syncthreads();
    if(t<256){
        const unsigned m=second[t],cnt=__popc(m);
        const unsigned prior=before[t>>3]+rze_scan<8>(cnt)-cnt;
        reinterpret_cast<uint2*>(first)[t]=rze_eight(in,int(before[32]),prior,m,stored);
    }
    __syncthreads();
    const unsigned m=first[t],cnt=__popc(m);
    const unsigned pos=block_prefix_sum<unsigned>(cnt,temp)-cnt;
    auto* out32=reinterpret_cast<unsigned*>(out)+8*t;
    #pragma unroll
    for(unsigned j=0;j<8;++j) {
        const unsigned k=4*j, nib=(m>>k)&15;
        unsigned v=0;
        if(nib){
            unsigned off=pos+__popc(m&((1u<<k)-1));
            #pragma unroll
            for(unsigned i=0;i<4;++i)if((nib>>i)&1) {
                if(off<stored)v|=unsigned(in[off])<<(i*8);
                ++off;
            }
        }
        out32[j]=v;
    }
}
