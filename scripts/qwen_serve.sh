#!/bin/bash
# Start the Qwen server (base weights; edits are loaded as LoRA adapters at runtime).
cd ~/wg-qwen
export VLLM_USE_FLASHINFER_SAMPLER=0   # no CUDA compiler on this machine; use the PyTorch sampler
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True   # edits are loaded and unloaded as adapters while the server runs
exec vllm-env/bin/vllm serve ~/wg-qwen/models/base --served-model-name base --port 8011 --host 127.0.0.1 \
  --dtype bfloat16 --max-model-len 40000 --gpu-memory-utilization 0.85 \
  --enable-lora --max-lora-rank 8 --max-loras ${WG_MAX_LORAS:-64} --max-cpu-loras 256 --max-num-seqs ${WG_MAX_SEQS:-72} --seed 0
