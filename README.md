# TurboQuant: FP8 KV Cache Inference for Finance LLMs — A Reference Deployment (2026)

> **Status — Decommissioned reference deployment.** The RunPod pod that backed
> this work was taken down due to provider-reliability constraints. The
> *technique* — FP8 KV cache via vLLM, with the documented config corrections —
> remains valid on any vLLM-capable provider. This repository is preserved as
> a reference implementation. The empirical results below (2× throughput on
> RTX A4000 at \$0.17/hr) were measured on the original deployment in
> April 2026 and are reported here as historical evidence, not as an active
> production claim.
>
> Cited in the [memory-oracle paper](https://github.com/ramene/memory-oracle/blob/main/paper/lncs/main.tex)
> §7 as the engineering pairing for Evidence-Bound Retrieval's longer
> amendment-merged contexts.

**2× inference throughput on consumer GPUs at \$0.17/hr** *(demonstrated, April 2026 reference deployment)*

This repo documents the deployment of a finance-trained 3B-parameter LLM
(Plutus-3B) with FP8 KV cache compression on RunPod, demonstrating 2× throughput
compared to standard inference — for less than the cost of a cup of coffee per day.

## The Journey

This repo documents the complete path from concept to working reference deployment:

1. **The Inspiration** — Nate Jones' research on LLM compute efficiency and inference optimization
2. **The Failed Attempts** — RunPod serverless (GPU shortage), Modal (priced out at $250/month), vLLM crashes (broken model configs)
3. **The Discoveries** — Two critical bugs in Plutus-3B's HuggingFace config that break vLLM on ANY provider
4. **The Solution** — RunPod pod + vLLM + FP8 KV cache + correct config = 2× throughput at $0.17/hr (deployment subsequently decommissioned; technique generalizes)

## What is TurboQuant?

TurboQuant uses vLLM's `--kv-cache-dtype fp8` flag to compress the key-value cache from FP16 to FP8 during inference. This halves the memory per token in the attention cache, which means:

- **2x more concurrent requests** in the same GPU memory
- **2x effective throughput** without any model quality degradation
- FP8 only affects the KV cache, not the model weights or computation

## The Problem We Solved

[Plutus-3B](https://huggingface.co/0xroyce/Plutus-3B) is a finance-trained LLM based on Llama 3.2 3B Instruct. It's designed for trading signal assessment — given a market signal, it returns a JSON decision (buy/sell/hold with confidence).

Running it on local hardware (Apple Silicon, single Ollama instance) caps at ~15 assessments/minute. For an autonomous trading platform with 5+ agents, each evaluating 15+ signals per cycle, that's a fatal bottleneck. Agents starve the position monitor, profitable exits are missed, money is lost.

## Critical Discovery: Broken HuggingFace Config

Plutus-3B's `config.json` on HuggingFace contains only:

```json
{"model_type": "llama"}
```

This is catastrophically incomplete. vLLM needs the full model configuration to load weights correctly. We discovered TWO bugs:

### Bug 1: Missing `architectures` field
vLLM throws `TypeError: 'NoneType' object is not iterable` when trying to identify the model architecture.

### Bug 2: Missing `vocab_size` field  
vLLM throws `AssertionError` on `vocab_parallel_embedding` when the embedding weights don't match the (guessed) vocabulary size.

### The Fix

The complete config.json for Plutus-3B (based on Llama 3.2 3B Instruct):

```json
{
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128009,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 3072,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 24,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "pad_token_id": 128004,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "vocab_size": 128256
}
```

This config is derived from [unsloth/Llama-3.2-3B-Instruct](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct) (the base model Plutus was fine-tuned from).

## Critical Discovery: Python Version Matters

- **Python 3.11** — vLLM 0.7.3 works correctly
- **Python 3.12** — vLLM 0.7.3 throws `architectures: NoneType` even with correct config.json

Use RunPod's `runpod-torch-v240` template (Python 3.10/3.11), NOT `runpod-torch-v280` (Python 3.12).

## Deployment Guide

### Prerequisites

- RunPod account with credits ($0.17/hr for RTX A4000)
- SSH public key added to RunPod Settings → SSH public keys

### Step 1: Create Pod

```bash
# Via RunPod API or web UI
# Template: runpod-torch-v240 (Python 3.11, CUDA 12.4)
# GPU: RTX A4000 16GB ($0.17/hr) — Plutus-3B needs ~9GB
# Ports: 8000/http, 22/tcp
# Disk: 30GB container + 30GB volume
```

### Step 2: SSH into Pod

RunPod pods using torch templates don't start sshd by default. Use RunPod's SSH proxy:

```bash
# Get your pod's host ID from the RunPod API or web UI
# Format: {podId}-{hash}
ssh -tt -i ~/.ssh/your_key {podHostId}@ssh.runpod.io
```

**Important:** The `-tt` flag forces PTY allocation, which RunPod's proxy requires. Without it you get `Error: Your SSH client doesn't support PTY`.

For piping commands (non-interactive):
```bash
echo 'your_command && exit' | ssh -tt -i ~/.ssh/your_key {podHostId}@ssh.runpod.io
```

### Step 3: Install vLLM + Download Model

```bash
pip install vllm==0.7.3 transformers==4.48.3 tokenizers==0.21.1 huggingface_hub hf_transfer

HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
import json, os

model_dir = '/workspace/plutus-3b'
os.makedirs(model_dir, exist_ok=True)

# Download model
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
```

### Step 4: Start vLLM with TurboQuant

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/plutus-3b \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 2048 \
  --dtype half \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90
```

The `--kv-cache-dtype fp8` flag is TurboQuant — FP8 KV cache compression for 2x throughput.

### Step 5: Test

```bash
# Via RunPod proxy URL
curl -X POST "https://{podId}-8000.proxy.runpod.net/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/plutus-3b",
    "prompt": "BTC-USDT at $71000. BULLISH 75%. JSON: {action,confidence,rationale}",
    "max_tokens": 100,
    "temperature": 0.2
  }'
```

Expected response (~3.5 seconds):
```json
{
  "choices": [{
    "text": "{\"action\": \"buy\", \"confidence\": 0.75, \"rationale\": \"Strong bullish momentum...\"}"
  }]
}
```

## Cost Comparison

| Setup | GPU | Cost | Throughput | TurboQuant |
|-------|-----|------|------------|------------|
| Local Ollama (Apple Silicon) | M-series | Free | ~15/min | No |
| RunPod Ollama (GGUF) | RTX A4000 | $0.17/hr | ~15/min | No |
| **RunPod vLLM + FP8** | **RTX A4000** | **$0.17/hr** | **~30/min** | **Yes** |
| Modal Serverless (vLLM) | T4 | $0.59/hr | ~20/min | Possible |
| RunPod A100 SXM | A100 80GB | $1.49/hr | ~60/min | Yes |

**Best value: RunPod RTX A4000 at $0.17/hr with FP8 KV cache = $4.08/day for 2x throughput.**

## Gotchas & Lessons Learned

### RunPod SSH
- Torch templates don't start sshd — use `ssh -tt {podHostId}@ssh.runpod.io`
- SSH keys must be added BEFORE pod creation
- The `-tt` flag is mandatory for the proxy

### Ollama on RunPod
- Default binds to `127.0.0.1` — proxy returns 502
- Must use `OLLAMA_HOST=0.0.0.0:11434 ollama serve`
- `ollama pull 0xroyce/plutus-3b` fails — use `hf.co/0xroyce/Plutus-3B:Q8_0`

### vLLM Config
- Plutus-3B's HuggingFace config.json is broken — must write full config
- `max_model_len` must be ≤ 2048 (model's max_position_embeddings)
- Python 3.12 breaks vLLM's architecture detection — use Python 3.11

### Modal vs RunPod
- Modal works but $250/month minimum for GPU scaling
- RunPod at $0.17/hr = $4.08/day = $122/month (48% cheaper)
- Both require the config.json fix

## Architecture: Dual-Inference for Trading

```
Agents (buy signals) ──→ RunPod vLLM + FP8 (TurboQuant)
                              ↓
                         30 assessments/min
                              
Monitor (sell signals) ──→ Local Ollama (dedicated)
                              ↓
                         15 assessments/min (no contention)
```

The key insight: buy-side and sell-side inference MUST be on separate endpoints. A single Plutus instance serving both leads to contention — agents starve the monitor, profitable exits are missed, money is lost.

## Files

- `config/plutus-3b-config.json` — The correct config.json for Plutus-3B
- `scripts/setup-runpod.sh` — Automated RunPod setup script
- `scripts/test-inference.sh` — Inference test script
- `docs/JOURNEY.md` — Full narrative from concept to deployment
- `docs/MODAL-NOTES.md` — Modal deployment notes (for reference)

## Pairing with Evidence-Bound Retrieval (EBR)

The technique TurboQuant validates — FP8 KV cache via vLLM — is the engineering
pairing referenced in the
[memory-oracle](https://github.com/ramene/memory-oracle) papers as the
cost-parity strategy for **longer amendment-merged retrieval contexts** (10–30 KB
vs. RAG's 2–5 KB). EBR's structural precedence invariant produces longer
prompts because amendment records are prepended to canonical content; FP8 KV
cache compression keeps per-token inference cost flat as those prompts grow.
See [`paper/lncs/main.tex`](https://github.com/ramene/memory-oracle/blob/main/paper/lncs/main.tex)
§7 for the full discussion. The technique generalizes — any vLLM-capable
provider can host the same configuration.

## Credits

- **Plutus-3B** by [0xroyce](https://huggingface.co/0xroyce/Plutus-3B) — finance-trained Llama 3.2 3B
- **vLLM** — high-throughput LLM serving with FP8 KV cache support
- **RunPod** — the GPU cloud provider used for the original reference deployment
- **Nate Jones** — inspiration on LLM compute efficiency

## License

MIT
