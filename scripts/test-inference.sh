#!/bin/bash
# Test TurboQuant inference endpoint
# Usage: ./test-inference.sh <runpod-pod-id>

POD_ID="${1:?Usage: ./test-inference.sh <pod-id>}"
URL="https://${POD_ID}-8000.proxy.runpod.net/v1/completions"

echo "Testing TurboQuant at: $URL"
echo ""

# Test 1: Model listing
echo "=== Models ==="
curl -s "$URL/../models" | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin)['data']]" 2>/dev/null || echo "Not ready"

echo ""

# Test 2: Finance assessment
echo "=== Finance Assessment ==="
time curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/plutus-3b",
    "prompt": "Signal: BTC-USDT at $71000. BULLISH momentum 75%. JSON ONLY: {action,confidence,entryPrice,stopLoss,takeProfit,rationale}",
    "max_tokens": 200,
    "temperature": 0.2
  }' | python3 -c "
import sys, json
d = json.load(sys.stdin)
text = d['choices'][0]['text']
print('Response:', text[:200])
print('Tokens:', d['usage']['completion_tokens'])
print('Prompt tokens:', d['usage']['prompt_tokens'])
" 2>/dev/null

echo ""
echo "=== TurboQuant Test Complete ==="
