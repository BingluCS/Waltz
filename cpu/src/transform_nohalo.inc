// Whole logical lines; no tiled loads or repeated lifting of overlap regions.
// Included inside transform.cpp's anonymous namespace after Simd definitions.
#if WALTZ_LINE_SCALAR
template<class T> struct ScalarLine {
    using V=T;static constexpr unsigned width=1;
    static V load(const T* p){return *p;}
    static void store(T* p,V x){*p=x;}
    static V set(T x){return x;}
    static V add(V x,V y){return x+y;}
    static V mul(V x,V y){return x*y;}
    static V fma(V x,V y,V z){return std::fma(x,y,z);}
};
#endif

template<class T,bool Inverse,class S>
WALTZ_LINE_TARGET void whole_line(T* scratch,T* const* dest,std::size_t stride,
 uint32_t n,int tile,bool folded,bool contiguous) {
    using V=typename S::V;using C=Constants<T>;constexpr unsigned W=S::width;
    const V a=S::set(Inverse?-C::a:C::a),b=S::set(Inverse?-C::b:C::b);
    const V g=S::set(Inverse?-C::g:C::g),d=S::set(Inverse?-C::d:C::d);
    const V kl=S::set(Inverse?C::ik:C::k),kh=S::set(Inverse?-C::k:-C::ik);
    for(uint32_t p=0;p<n;++p){
        size_t at=(Inverse?(p/2+(p%2?half(n):0)):p)*stride;
        V v;
        if(contiguous)v=S::load(dest[0]+at);
        else {T values[W];for(unsigned l=0;l<W;++l)values[l]=dest[l][at];v=S::load(values);}
        if constexpr(Inverse)v=S::mul(v,p%2?kh:kl);
        S::store(scratch+size_t(p)*W,v);
    }
    auto stage=[&](unsigned parity,V c) WALTZ_LINE_TARGET {
        for(uint32_t p=parity;p<n;p+=2){
            uint32_t lo=p?p-1:1,hi=p+1<n?p+1:n-2;
            S::store(scratch+size_t(p)*W,S::fma(c,
                S::add(S::load(scratch+size_t(lo)*W),S::load(scratch+size_t(hi)*W)),
                S::load(scratch+size_t(p)*W)));
        }
    };
    T* fix=scratch+size_t(n)*W;
    auto read=[&](int64_t p) WALTZ_LINE_TARGET {return S::load(scratch+size_t(simd_address<false>(p,n))*W);};
    auto adjust=[&](unsigned phase) WALTZ_LINE_TARGET {
        size_t f=0;
        for(uint32_t start=0;start<n;start+=unsigned(tile),f+=3*W){
            const int64_t s=start,r=int64_t(start)+tile;
            V left,right;
            if(phase==0){
                auto raw=[&](int64_t p) WALTZ_LINE_TARGET {
                    size_t at=simd_address<true>(p,n)*stride;
                    if(contiguous)return S::load(dest[0]+at);
                    T v[W];for(unsigned l=0;l<W;++l)v[l]=dest[l][at];return S::load(v);
                };
                left=S::fma(d,S::fma(raw(s-3),kh,read(s-1)),read(s-2));
                right=S::fma(d,S::fma(raw(r+3),kh,read(r+1)),read(r+2));
            }else{
                left=S::load(fix+f);right=S::load(fix+f+W);
                if(phase==1){
                    left=S::fma(g,S::add(left,read(s)),read(s-1));
                    right=S::fma(g,S::add(read(r),right),read(r+1));
                }else if(phase==2){
                    left=S::fma(b,S::add(left,read(s+1)),read(s));
                    right=S::fma(b,S::add(read(r-1),right),read(r));
                }else if(phase==3){
                    right=S::fma(a,S::add(read(r-2),right),read(r-1));
                    S::store(fix+f+2*W,S::fma(a,S::add(left,read(s+2)),read(s+1)));
                }else{
                    S::store(scratch+size_t(start)*W,left);
                    if(start+1<n)S::store(scratch+size_t(start+1)*W,S::load(fix+f+2*W));
                    if(r-1<n)S::store(scratch+size_t(r-1)*W,right);
                }
            }
            S::store(fix+f,left);S::store(fix+f+W,right);
        }
    };
    if(n==1){
        // Reflecting a one-sample axis aliases both parity inputs to sample 0.
        V e=S::load(scratch),o;
        if constexpr(Inverse){
            T values[W];for(unsigned l=0;l<W;++l)values[l]=dest[l][0];
            const V raw=S::load(values);
            o=S::mul(raw,kh);
            const V center=S::fma(d,S::add(o,o),e);
            // The folded GPU tile contracts the left neighbor at position -2.
            // All other neighbors reflect to the same sample, without contraction.
            const V left=folded?S::fma(d,S::fma(raw,kh,o),e):center;
            const V odd_left=S::fma(g,S::add(left,center),o);
            const V odd_right=S::fma(g,S::add(center,center),o);
            e=S::fma(b,S::add(odd_left,odd_right),center);
        }else{
            o=e;o=S::fma(a,S::add(e,e),o);e=S::fma(b,S::add(o,o),e);
            o=S::fma(g,S::add(e,e),o);e=S::fma(d,S::add(o,o),e);e=S::mul(e,kl);
        }
        S::store(scratch,e);
    }else{
        if constexpr(Inverse){
            if(folded)adjust(0);
            stage(0,d);if(folded)adjust(1);
            stage(1,g);if(folded)adjust(2);
            stage(0,b);if(folded)adjust(3);
            stage(1,a);if(folded)adjust(4);
        }
        else {stage(1,a);stage(0,b);stage(1,g);stage(0,d);}
    }
    for(uint32_t p=0;p<n;++p){
        V v=S::load(scratch+size_t(p)*W);
        if constexpr(!Inverse)if(n!=1)v=S::mul(v,p%2?kh:kl);
        size_t at=(Inverse?p:(p/2+(p%2?half(n):0)))*stride;
        if(contiguous)S::store(dest[0]+at,v);
        else {T values[W];S::store(values,v);for(unsigned l=0;l<W;++l)dest[l][at]=values[l];}
    }
}

#if WALTZ_LINE_SCALAR
template<class T,bool Inverse>
void axis_whole_scalar(T* data,T* scratch,Shape full,Shape active,Axis axis,int tile,size_t begin=0){
    const uint32_t n=length(active,axis);const size_t stride=line_stride(full,axis),count=line_count(active,axis);
    for(size_t line=begin;line<count;++line){
        T* dest[]={data+line_offset(full,active,axis,line)};
        whole_line<T,Inverse,ScalarLine<T>>(scratch,dest,stride,n,tile,axis!=X,false);
    }
}
#elif WALTZ_CPU_X86_TARGET
template<class T,bool Inverse>
WALTZ_CPU_AVX2 void axis_whole_simd(T* data,T* scratch,Shape full,Shape active,Axis axis,int tile){
    using S=Simd<T>;constexpr unsigned W=S::width;
    const uint32_t n=length(active,axis);const size_t stride=line_stride(full,axis),count=line_count(active,axis);
    size_t line=0;
    for(;count-line>=W;line+=W){
        T* dest[W];for(unsigned l=0;l<W;++l)dest[l]=data+line_offset(full,active,axis,line+l);
        bool contiguous=axis!=X&&line%active.nx+W<=active.nx;
        whole_line<T,Inverse,S>(scratch,dest,stride,n,tile,axis!=X,contiguous);
    }
    axis_whole_scalar<T,Inverse>(data,scratch,full,active,axis,tile,line);
}
#endif
