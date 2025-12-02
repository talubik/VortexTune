#!/usr/bin/env bash

cores="$1"
warps="$2"
threads="$3"
clusters="$4"
l2cache="$5"
l3cache="$6"
vortex_dir="$7"
spla_dir="$8"
mtx_path="$9"

cd "$spla_dir"
source "$vortex_dir/build/ci/toolchain_env.sh"
eval "$(python3 configure_vortex.py --vortex-dir="$vortex_dir"  --cores=$cores --warps=$warps --threads=$threads --clusters=$clusters --driver=simx --l2cache=$l2cache --l3cache=$l3cache)"
"$spla_dir/build/tc" --mtxpath="$mtx_path" --run-cpu=False --niters=1 --platform=0 --device=0 --run-ref=False
# "$spla_dir/build/tests/test_vector" --gtest_filter=vector.reduce_perf