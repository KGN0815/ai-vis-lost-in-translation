#!/usr/bin/env bash
# Run with the project's Intel-enabled PyTorch and optional local Fedora runtime.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$project_dir/.venv/intel-runtime/usr/lib64"
if [[ -d "$runtime_dir" ]]; then
    export LD_LIBRARY_PATH="$runtime_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [[ -f "$runtime_dir/intel-opencl/libigdrcl.so" ]]; then
        # oneDNN's attention backend also needs the matching OpenCL device.
        icd_dir="$project_dir/data/cache/OpenCL/vendors"
        mkdir -p "$icd_dir"
        printf '%s\n' "$runtime_dir/intel-opencl/libigdrcl.so" > "$icd_dir/intel.icd"
        export OCL_ICD_VENDORS="$icd_dir"
    fi
fi
export HF_HOME="${HF_HOME:-$project_dir/data/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$project_dir/data/cache}"
cd "$project_dir"
exec "$project_dir/.venv/bin/python" -m implementation.embed --device xpu "$@"
