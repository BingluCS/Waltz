// Decode the packed sign stream. Each lane owns 16 consecutive coefficients;
// a prefix over lanes gives the position of its first nonzero's sign bit.
// raw and BIT_2/RZE chunks both supply the same decoded magnitude array.
__device__ __forceinline__ void expand_sign_groups(const uint16_t* mag,
    unsigned elements,const byte* packed,unsigned bytes,byte* temp) {
    static_assert(ITEMS_PER_THREAD == 16 && TPB*16 == ELEMS_PER_CHUNK);
    const unsigned base=threadIdx.x*16;
    unsigned nz=0;
    if(base+16<=elements){
        #pragma unroll
        for(unsigned j=0;j<16;++j)nz|=unsigned(mag[base+j]!=0)<<j;
    }else{
        for(unsigned j=0;base+j<elements;++j)nz|=unsigned(mag[base+j]!=0)<<j;
    }
    const unsigned count=__popc(nz);
    const unsigned rank=block_prefix_sum<unsigned>(count,temp)-count;
    __syncthreads(); // All prefix consumers are done before reusing temp.
    unsigned neg=0;
    if(nz){
        const unsigned start=rank>>3,shift=rank&7;
        unsigned bits=start<bytes?packed[start]:0;
        if(shift+count>8 && start+1<bytes)bits|=unsigned(packed[start+1])<<8;
        if(shift+count>16 && start+2<bytes)bits|=unsigned(packed[start+2])<<16;
        bits>>=shift;
        // The common cases cost no deposit loop: no negative or all nonzero.
        bits &= (1u<<count)-1;
        if(nz==65535)neg=bits;
        else while(bits){const unsigned b=nz&-nz;if(bits&1)neg|=b;bits>>=1;nz-=b;}
    }
    reinterpret_cast<uint16_t*>(temp)[threadIdx.x]=uint16_t(neg);
    __syncthreads();
}
