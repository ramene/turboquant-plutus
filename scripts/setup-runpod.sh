#!/bin/bash
# TurboQuant: Setup vLLM + FP8 KV Cache on RunPod
# Template: runpod-torch-v240 (Python 3.11 — REQUIRED, 3.12 breaks vLLM)
# GPU: RTX A4000 16GB ($0.17/hr) or any 16GB+ NVIDIA GPU
#
# Usage: pipe this script through SSH
#   cat scripts/setup-runpod.sh | ssh -tt -i ~/.ssh/your_key {podHostId}@ssh.runpod.io

set -e

echo "=== TurboQuant Setup ==="
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "Python:"
python3 --version

echo "=== Installing vLLM ==="
pip install vllm==0.7.3 transformers==4.48.3 tokenizers==0.21.1 huggingface_hub hf_transfer 2>&1 | tail -3

echo "=== Downloading Plutus-3B ==="
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
import json, os

model_dir = '/workspace/plutus-3b'
os.makedirs(model_dir, exist_ok=True)

print('Downloading from HuggingFace...')
snapshot_download('0xroyce/Plutus-3B', local_dir=model_dir)

# Write CORRECT config.json (the HuggingFace one is broken)
config = {
    'architectures': ['LlamaForCausalLM'],
    'vocab_size': 128256,
    'hidden_size': 3072,
    'num_hidden_layers': 28,
    'num_attention_heads': 24,
    'num_key_value_heads': 8,
    'intermediate_size': 8192,
    'head_dim': 128,
    'hidden_act': 'silu',
    'max_position_embeddings': 131072,
    'rms_norm_eps': 1e-05,
    'rope_theta': 500000.0,
    'rope_scaling': {
        'factor': 32.0,
        'high_freq_factor': 4.0,
        'low_freq_factor': 1.0,
        'original_max_position_embeddings': 8192,
        'rope_type': 'llama3'
    },
    'tie_word_embeddings': True,
    'torch_dtype': 'bfloat16',
    'use_cache': True,
    'model_type': 'llama',
    'attention_bias': False,
    'attention_dropout': 0.0,
    'mlp_bias': False,
    'bos_token_id': 128000,
    'eos_token_id': 128009,
    'pad_token_id': 128004,
    'pretraining_tp': 1,
    'initializer_range': 0.02,
}
with open(os.path.join(model_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)
print('Model + config ready')
"

echo "=== Starting vLLM with TurboQuant (FP8 KV Cache) ==="
python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/plutus-3b \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 2048 \
  --dtype half \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90
