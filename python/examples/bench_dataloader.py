"""Compare CPU decode plus H2D with native GPU decode on the same HDF5 samples."""
import argparse
from statistics import median
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from pywaltz.dataloader import WaltzDataset
from pywaltz.gpu_dataloader import WaltzGPUDataLoader


def epoch(loader, device, copy_to_device):
    torch.cuda.synchronize(device)
    start = perf_counter()
    size = 0
    for x in loader:
        if copy_to_device:
            x = x.to(device, non_blocking=True)
        size += x.numel() * x.element_size()
    torch.cuda.synchronize(device)
    return size, perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--key", default="fields")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--baseline", choices=("cpu","serial"), default="cpu",
                   help="compare against CPU + transfer or the previous per-chunk GPU decoder")
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be positive")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    opts = dict(batch_size=args.batch_size, num_workers=args.workers,
                pin_memory=True, persistent_workers=bool(args.workers))
    on_cpu = args.baseline=="cpu"
    cpu = (DataLoader(WaltzDataset(args.file, args.key),
                      multiprocessing_context="spawn" if args.workers else None, **opts)
           if on_cpu else WaltzGPUDataLoader(args.file,args.key,device=device,decode_mode="serial",**opts))
    gpu = WaltzGPUDataLoader(args.file, args.key, device=device, decode_mode="batch", **opts)
    jobs = ((cpu, True), (gpu, False))
    size, _ = epoch(cpu, device, on_cpu)
    if epoch(gpu, device, False)[0] != size:
        raise RuntimeError("different decoded byte counts")
    seconds = {True: [], False: []}
    for i in range(args.repeats):
        for loader, copy in (jobs if i % 2 == 0 else jobs[::-1]):
            n, t = epoch(loader, device, copy and on_cpu)
            if n != size:
                raise RuntimeError("decoded byte count changed")
            seconds[copy].append(t)
    print(f"{torch.cuda.get_device_name(device)}, {size} decoded bytes/epoch")
    for copy, label in ((True, "CPU decode + H2D" if on_cpu else "Serial GPU decode"), (False, "Batched GPU decode")):
        print(f"{label}: {size/median(seconds[copy])/1e9:.3f} GB/s, epoch seconds: {seconds[copy]}")
    print(f"Loader speedup: {median(seconds[True])/median(seconds[False]):.2f}x")


if __name__ == "__main__":
    main()
