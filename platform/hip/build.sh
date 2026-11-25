#!/usr/bin/env bash
set -eo pipefail
src=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$src/../.." && pwd)
arch=${1:-gfx936}
out=${2:-"$src/build-$arch"}

if ! command -v hipcc >/dev/null && [[ -f /opt/dtk/env.sh ]]; then
    source /opt/dtk/env.sh >/dev/null
fi
if ! command -v hipcc >/dev/null; then
    printf 'hipcc not found; load your HIP/DCU SDK environment first.\n' >&2
    exit 1
fi
set -u
native_flags=()
if [[ ${WALTZ_HIP_TRACE:-0} == 1 ]]; then native_flags+=(-DWALTZ_HIP_TRACE); fi
python3 "$src/generate.py" "$root" "$out/generated"
gccdir=$(dirname "$(g++ -print-libgcc-file-name)")
hipcc --gcc-install-dir="$gccdir" --offload-arch="$arch" -std=c++17 -O3 -ffp-contract=off -fno-fast-math \
    -fdenormal-fp-math=ieee -Wno-return-type -fPIC -I"$out/generated" -I"$root/include" -I"$root/include/lossless/lc" \
    "${native_flags[@]}" -x hip -c "$src/native.cu" -o "$out/native.o"
hipcc --gcc-install-dir="$gccdir" --offload-arch="$arch" -shared "$out/native.o" -o "$out/libwaltz_hip_native.so"
