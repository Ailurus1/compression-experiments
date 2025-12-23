#!/bin/bash

python3 -m venv compression-experiments-venv
source ./compression-experiments-venv/bin/activate

python3 -m pip install -r requirements.txt
# Unfortunately, llm-compressor is not compatible with latest vllm right now
# due to mismatch in compressed-tensors version
# However, it's still working way to manually upgrade the package
python3 -m pip install --upgrade compressed-tensors
