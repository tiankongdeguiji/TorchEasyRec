#!/bin/bash
# One-time setup on tianyi_l20n_llm4rec_serve (run ON the remote host).
# Checks out the ty_git bench branches, installs sid-gr-inference (no-deps, preserving
# the machine's newer kernel-wheel versions), validates imports, downloads the model.
set -exo pipefail

BASE=/mnt/data/hongsheng.jhs
RECSYS=$BASE/recsys-examples
SIDGR=$RECSYS/examples/sid-gr-inference

# 1. bench branches from ty_git (repos already have the ty_git remote)
git -C $BASE/sglang fetch ty_git llm4rec/beam-search-bench
git -C $BASE/sglang checkout -B llm4rec/beam-search-bench FETCH_HEAD
git -C $RECSYS fetch ty_git llm4rec/sidgr-bench
git -C $RECSYS checkout -B llm4rec/sidgr-bench FETCH_HEAD
git -C $BASE/TorchEasyRec fetch ty_git rfc_llm4rec_inference_bench
git -C $BASE/TorchEasyRec checkout -B rfc_llm4rec_inference_bench FETCH_HEAD

# 2. sid-gr-inference: editable, WITHOUT the [kernels] extra (would downgrade
#    cutlass-dsl 4.5.2 -> 4.5.1 etc. on this shared env); core dep huggingface-hub
#    is already present.
pip install -e "$SIDGR" --no-deps

# 3. import validation (targeted installs only if something is missing)
export GR_DECODE_ATTEN_ROOT=$RECSYS/corelib/gr_decode_atten
python - <<'PY'
import importlib, subprocess, sys

missing = []
for mod, pipname in [
    ("torch", None), ("flash_attn", None), ("flashinfer", None),
    ("cutlass", None), ("quack", None), ("cuda.bindings.driver", "cuda-python"),
    ("tvm_ffi", "apache-tvm-ffi"), ("pytest", "pytest"),
]:
    try:
        importlib.import_module(mod)
        print("ok:", mod)
    except Exception as e:
        print("MISSING:", mod, "->", e)
        if pipname:
            missing.append(pipname)
if missing:
    print("installing:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

import os, sys
sys.path.insert(0, os.environ["GR_DECODE_ATTEN_ROOT"])
import interface
assert hasattr(interface, "beam_decode_atten") or hasattr(interface, "beam_decode_attn"), dir(interface)
print("ok: gr_decode_atten interface")
PY

# 4. model via hf-mirror (huggingface.co is blocked from this pod)
MODEL_DIR=$BASE/models/Qwen3-1.7B
if [ ! -f "$MODEL_DIR/config.json" ]; then
  export HF_ENDPOINT=https://hf-mirror.com
  (hf download Qwen/Qwen3-1.7B --local-dir "$MODEL_DIR" \
     || huggingface-cli download Qwen/Qwen3-1.7B --local-dir "$MODEL_DIR")
fi
ls -la "$MODEL_DIR" | head
echo SETUP_DONE
