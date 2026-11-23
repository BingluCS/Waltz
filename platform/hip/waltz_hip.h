#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct waltz_hip_session waltz_hip_session;
typedef struct {uint32_t x,y,z;int width;double bound;} waltz_hip_config;
waltz_hip_session* waltz_hip_create(int device);
void waltz_hip_destroy(waltz_hip_session*);
const char* waltz_hip_error(const waltz_hip_session*);
size_t waltz_hip_max_bytes(const waltz_hip_config*);
/* All payload pointers are device pointers. The synchronous calls read a
 * small header/status on the host to report lengths and errors. One stream
 * and one caller per session; buffers are non-overlapping. Decode is for
 * trusted WLT2 mode-2 streams only, not an untrusted-file parser. */
int waltz_hip_encode(waltz_hip_session*,const waltz_hip_config*,const void*,void*,size_t,size_t*);
int waltz_hip_decode(waltz_hip_session*,const waltz_hip_config*,const void*,size_t,void*,size_t);
/* Optional libwaltz_hip_waw2.so. Same device-pointer API; native_cap bounds
 * the intermediate WLT2 archive. No CPU entropy or host payload fallback. */
typedef struct waltz_hip_waw2 waltz_hip_waw2;
size_t waltz_hip_waw2_capacity(size_t native_cap);
waltz_hip_waw2* waltz_hip_waw2_create(int device,size_t native_cap);
void waltz_hip_waw2_destroy(waltz_hip_waw2*);
const char* waltz_hip_waw2_error(const waltz_hip_waw2*);
int waltz_hip_waw2_encode(waltz_hip_waw2*,const waltz_hip_config*,const void*,void*,size_t,size_t*);
int waltz_hip_waw2_decode(waltz_hip_waw2*,const waltz_hip_config*,const void*,size_t,void*,size_t);
#ifdef __cplusplus
}
#endif
