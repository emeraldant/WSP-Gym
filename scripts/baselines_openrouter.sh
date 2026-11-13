#!/bin/bash
# Off-the-shelf baselines through OpenRouter: every model on the first 100 frozen test_id episodes, in both
# regimes (no reasoning; with reasoning). The model list and the rules were fixed before any of it ran
# (NOTES.md, Nov 13). Resumable: episodes already answered are skipped.
#
#   scripts/baselines_openrouter.sh            run everything
#   scripts/baselines_openrouter.sh credits    print the key's usage and limit as OpenRouter reports them
#
# The key is read from WEIGHTGYM_API_KEY or from ~/.openrouter_key and is never printed or logged.
cd "$(dirname "$0")/.."
: "${WEIGHTGYM_API_KEY:=$(cat ~/.openrouter_key 2>/dev/null | tr -d '[:space:]')}"
[ -n "$WEIGHTGYM_API_KEY" ] || { echo "no key: put it in ~/.openrouter_key or set WEIGHTGYM_API_KEY"; exit 1; }
export WEIGHTGYM_API_KEY PYTHONPATH=. WEIGHTGYM_BUDGET_USD=${WEIGHTGYM_BUDGET_USD:-60}
PY=.venv/bin/python

if [ "$1" = credits ]; then
  curl -s -m 30 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $WEIGHTGYM_API_KEY" |
    $PY -c "import json,sys; d=json.load(sys.stdin).get('data',{}); print({k: d.get(k) for k in ('usage','limit','limit_remaining')})"
  exit
fi

mkdir -p runs/openrouter_logs
one() {  # one <arm prefix> <model id> [flags for the reasoning arm]
  $PY -m weightgym.frontier_eval --model "$2" --mode direct    --n 100 --concurrency 6 --arm "$1_direct"       >> "runs/openrouter_logs/$1.log" 2>&1
  $PY -m weightgym.frontier_eval --model "$2" --mode reasoning --n 100 --concurrency 6 --arm "$1_reasoning" $3 >> "runs/openrouter_logs/$1.log" 2>&1
}
# Closed models already in the table: finish the same 100 episodes with the settings they started with.
one frontier_sol      openai/gpt-6.1-sol &
one frontier_astra    openai/gpt-6-astra &
# Closed models from three other labs.
one frontier_fable    anthropic/claude-fable-5.2 --reasoning-on &
one frontier_gemini   google/gemini-4-carbon --reasoning-on &
one frontier_grok     x-ai/grok-4.7 --reasoning-on &
# Open-weights models (each has a Hugging Face id in OpenRouter's model list).
one open_inkling1     thinkingmachines/inkling --reasoning-on &
one open_qwen         qwen/qwen3.8-2.4t-a95b --reasoning-on &
one open_deepseek     deepseek/deepseek-v4-pro-0813 --reasoning-on &
one open_kimi         moonshotai/kimi-k3.1-pro --reasoning-on &
one open_glm          z-ai/glm-5.3 --reasoning-on &
one open_muse         meta/muse-spark-open --reasoning-on &
one open_mistral      mistralai/mistral-large-4-0 --reasoning-on &
one open_nemotron     nvidia/nemotron-3-ultra-550b-a55b --reasoning-on &
one open_beam         reflection/beam-1.0-501b-a23b --reasoning-on &
wait
tail -q -n 2 runs/openrouter_logs/*.log
