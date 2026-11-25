"""Read independent volumes, or ordered input/target frames with --lead."""

import argparse

from torch.utils.data import DataLoader

from pywaltz.dataloader import WaltzDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--key", default="fields")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lead", type=int, default=0)
    p.add_argument("--device-decode", metavar="DEVICE",
                   help="decode chunks on cuda:0 (NVIDIA or HIP/DCU) instead of CPU")
    args = p.parse_args()
    if args.device_decode:
        from pywaltz.gpu_dataloader import WaltzGPUDataLoader
        loader = WaltzGPUDataLoader(args.file, key=args.key, lead=args.lead,
                                    device=args.device_decode, batch_size=args.batch_size,
                                    num_workers=args.workers)
    else:
        ds = WaltzDataset(args.file, key=args.key, lead=args.lead)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    count = 0
    for batch in loader:
        x = batch[0] if args.lead else batch
        if count == 0:
            print("inputs:", tuple(x.shape), x.dtype, x.device)
            if args.lead:
                print("targets:", tuple(batch[1].shape))
        count += len(x)
    print(f"read {count} samples")


if __name__ == "__main__":
    main()
