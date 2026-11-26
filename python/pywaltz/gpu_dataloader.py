"""Read trusted Waltz HDF5 chunks on CPU; reconstruct directly on CUDA/HIP."""
import ctypes as C
import os
import math
from functools import partial

import torch
from torch.utils.data import DataLoader, get_worker_info

from ._raw_chunks import RawFrames, pack


class _HipConfig(C.Structure):
    _fields_ = [("x", C.c_uint32), ("y", C.c_uint32), ("z", C.c_uint32),
                ("width", C.c_int), ("bound", C.c_double)]


class _BatchShape(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
                ("x","y","z","fx","fy","fz","samples","sides","width")]


class _Decoder:
    def __init__(self, dataset, device, library, mode="auto"):
        self.device = device
        self.mode = mode
        self.hip = bool(torch.version.hip)
        self.session = None
        var = "WALTZ_HIP_LIBRARY" if self.hip else "WALTZ_CUDA_LIBRARY"
        name = "libwaltz_hip_native.so" if self.hip else "libwaltz.so"
        try:
            self.lib = C.CDLL(os.fspath(library or os.getenv(var) or name))
            p, sz = C.c_void_p, C.c_size_t
            self.multi = (self._bind("waltz_gpu_decode_batch", C.c_char_p,
                          [C.POINTER(_BatchShape),p,p,sz,p,sz,p,sz,C.c_double,p])
                          if hasattr(self.lib,"waltz_gpu_decode_batch") else None)
            if mode == "batch":
                if self.multi is None:
                    raise RuntimeError("batch decoder unavailable; rebuild the native Waltz library")
                return  # No single-chunk entry points are needed in this mode.
            if self.hip:
                self.config = _HipConfig(*dataset.xyz, dataset.width, dataset.bound)
                self._bind("waltz_hip_create", p, [C.c_int])
                self._bind("waltz_hip_destroy", None, [p])
                self._bind("waltz_hip_error", C.c_char_p, [p])
                self.decode = self._bind("waltz_hip_decode", C.c_int,
                                         [p, C.POINTER(_HipConfig), p, sz, p, sz])
            else:
                self.decode = self._bind("waltz_cuda_decode_chunk", C.c_char_p,
                                         [C.c_uint32]*3 + [C.c_int, p, sz, p, sz, p])
        except (OSError, AttributeError) as exc:
            raise RuntimeError(f"build the matching Waltz backend and set {var}: {exc}") from exc

    def _bind(self, name, result, args):
        fn = getattr(self.lib, name)
        fn.restype, fn.argtypes = result, args
        return fn

    def close(self):
        if self.session is not None:
            self.lib.waltz_hip_destroy(self.session)
            self.session = None

    def tile_into(self, d, ptr, size, out, stream):
        args = (ptr, size, out.data_ptr(), out.numel()*d.width)
        if self.hip:
            if self.session is None:
                self.session = self.lib.waltz_hip_create(self.device.index)
                if not self.session:
                    raise RuntimeError("cannot create Waltz HIP wave64 decoder")
            rc = self.decode(self.session, C.byref(self.config), *args)
            error = self.lib.waltz_hip_error(self.session) if rc else None
        else:
            error = self.decode(*d.xyz, d.width, *args, stream.cuda_stream)
        if error is not None:
            raise RuntimeError(f"Waltz device decode: {error.decode(errors='replace')}")

    def batch(self, d, batch):
        caller = torch.cuda.current_stream(self.device)
        jobs = len(batch.spans)
        # Three equally sized device work banks; avoid tripling a large
        # training batch's memory footprint. Serial mode needs just one tile.
        stride = (math.prod(d.tile)+3) & ~3
        scratch_bytes = 3*stride*jobs*d.width + 8*256*32*4 + 16
        fits = jobs<=65535 and max(d.tile)<=65535 and scratch_bytes<=256*1024**2
        if self.mode == "batch" and not (fits and batch.indexed):
            raise ValueError("batch exceeds GPU workspace/launch limits or lacks span metadata")
        if self.mode != "serial" and batch.indexed and self.multi is not None and fits:
            sides = 2 if d.lead else 1
            shape = _BatchShape(*d.xyz, *(*reversed(d.shape),1)[:3], batch.count, sides, d.width)
            dtype = torch.float32 if d.width==4 else torch.float64
            # Current torch stream for both HIP and CUDA. All allocations are
            # owned by torch, not shared between decoders or devices.
            packed = batch.payload.to(self.device, non_blocking=True)
            scratch = torch.empty(scratch_bytes,dtype=torch.uint8,device=self.device)
            out = torch.empty((sides,batch.count,*d.shape),dtype=dtype,device=self.device)
            error = self.multi(C.byref(shape),batch.payload.data_ptr(),packed.data_ptr(),packed.numel(),
                               scratch.data_ptr(),scratch.numel(),out.data_ptr(),out.numel()*d.width,
                               d.fill,caller.cuda_stream)
            if error is not None:
                raise RuntimeError(f"Waltz batch decode: {error.decode(errors='replace')}")
            return tuple(out.unbind(0)) if d.lead else out[0]
        # HIP's native C API uses the default stream. Keep upload and assembly
        # there too and publish the outputs to the caller's actual stream.
        stream = torch.cuda.default_stream(self.device) if self.hip else caller
        dtype = torch.float32 if d.width == 4 else torch.float64
        with torch.cuda.stream(stream):
            packed = batch.payload.to(self.device, non_blocking=True)
            outputs = [torch.empty((batch.count, *d.shape), dtype=dtype, device=self.device)
                       for _ in range(2 if d.lead else 1)]
            spans = iter(batch.spans)
            scratch = None
            for i in range(batch.count):
                for out in outputs:
                    for at in d.origins():
                        offset, size = next(spans)
                        extent = tuple(min(c, n-a) for n,c,a in zip(d.shape, d.tile, at))
                        region = tuple(slice(a, a+e) for a,e in zip(at, extent))
                        dst = out[(i, *region)]
                        if not size:
                            dst.fill_(d.fill)
                            continue
                        direct = extent == d.tile and dst.is_contiguous() and dst.data_ptr() % 16 == 0
                        if not direct and scratch is None:
                            scratch = torch.empty(d.tile, dtype=dtype, device=self.device)
                        self.tile_into(d, packed.data_ptr()+offset, size, dst if direct else scratch, stream)
                        if not direct:
                            dst.copy_(scratch[tuple(slice(e) for e in extent)])
        if stream != caller:
            caller.wait_stream(stream)
            for out in outputs:
                out.record_stream(caller)
        return tuple(outputs) if d.lead else outputs[0]


class WaltzGPUDataLoader:
    """Yield device tensors from trusted pywaltz.hdf5 files without CPU decoding.

    Use one training process per accelerator. A positive lead yields
    (input, target); the default yields one tensor per batch.
    """
    def __init__(self, path, key="fields", *, device="cuda", library=None, lead=0,
                 batch_size=1, num_workers=0, pin_memory=True, shuffle=False,
                 sampler=None, drop_last=False, persistent_workers=False, decode_mode="auto"):
        if decode_mode not in ("auto","batch","serial"):
            raise ValueError("decode_mode must be auto, batch or serial")
        if batch_size is None:
            raise ValueError("an integer batch_size is required")
        self.decode_mode = decode_mode
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("use a torch 'cuda' device for NVIDIA CUDA or HIP/DCU")
        self.library = library
        self.dataset = RawFrames(path, key, lead)
        self._io = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=shuffle, sampler=sampler, drop_last=drop_last,
                              collate_fn=partial(pack,indexed=decode_mode!="serial"), pin_memory=pin_memory,
                              multiprocessing_context="spawn" if num_workers else None,
                              persistent_workers=persistent_workers)

    def __len__(self):
        return len(self._io)

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("iterate the GPU DataLoader in the training process")
        device = self.device
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        with torch.cuda.device(device):
            decoder = _Decoder(self.dataset, device, self.library, self.decode_mode)
        try:
            with torch.cuda.device(device):
                batches = iter(self._io)
            for batch in batches:
                with torch.cuda.device(device):
                    result = decoder.batch(self.dataset, batch)
                yield result
        finally:
            with torch.cuda.device(device):
                decoder.close()
