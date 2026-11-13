#!/bin/bash
# Off-the-shelf baselines through OpenRouter: every model on the first 100 frozen test_id episodes, in both
# regimes (no reasoning; with reasoning). The model list and the rules were fixed before any of it ran
# (Nov 13, 2026). Resumable: episodes already answered are skipped.
#
#   scripts/baselines_openrouter.sh            run everything, cheapest arms first
#   scripts/baselines_openrouter.sh credits    print the key's usage and limit as OpenRouter reports them
#
# The key is read from WEIGHTGYM_API_KEY or from ~/.openrouter_key and is never printed or logged.
cd "$(dirname "$0")/.."
: "${WEIGHTGYM_API_KEY:=$(cat ~/.openrouter_key 2>/dev/null | tr -d '[:space:]')}"
[ -n "$WEIGHTGYM_API_KEY" ] || { echo "no key: put it in ~/.openrouter_key or set WEIGHTGYM_API_KEY"; exit 1; }
export WEIGHTGYM_API_KEY PYTHONPATH=.
PY=.venv/bin/python

if [ "$1" = credits ]; then
  curl -s -m 30 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $WEIGHTGYM_API_KEY" |
    $PY -c "import json,sys; d=json.load(sys.stdin).get('data',{}); print({k: d.get(k) for k in ('usage','limit','limit_remaining')})"
  exit
fi

mkdir -p runs/openrouter_logs
# Spend stop in dollars, read again before each arm starts, so it can be changed between stages without a restart.
budget() { cat runs/openrouter_budget 2>/dev/null || echo 60; }
d() {  # d <arm prefix> <model id>: the no-reasoning arm
  WEIGHTGYM_BUDGET_USD=$(budget) $PY -m weightgym.frontier_eval --model "$2" --mode direct --n 100 --concurrency 8 --arm "$1_direct" >> "runs/openrouter_logs/$1_direct.log" 2>&1
}
r() {  # r <arm prefix> <model id> [flags]: the reasoning arm
  WEIGHTGYM_BUDGET_USD=$(budget) $PY -m weightgym.frontier_eval --model "$2" --mode reasoning --n 100 --concurrency 20 --arm "$1_reasoning" $3 >> "runs/openrouter_logs/$1_reasoning.log" 2>&1
}
ON=--reasoning-on
# Closed: Sol and Astra are already in the table and keep the settings they started with; three more labs follow.
# Open-weights: each has a Hugging Face id in OpenRouter's model list.
ALL="frontier_sol=openai/gpt-6.1-sol frontier_astra=openai/gpt-6-astra frontier_fable=anthropic/claude-fable-5.2
     frontier_gemini=google/gemini-4-carbon frontier_grok=x-ai/grok-4.7 open_inkling1=thinkingmachines/inkling
     open_qwen=qwen/qwen3.8-2.4t-a95b open_deepseek=deepseek/deepseek-v4-pro-0813 open_kimi=moonshotai/kimi-k3.1-pro
     open_glm=z-ai/glm-5.3 open_muse=meta/muse-spark-open open_mistral=mistralai/mistral-large-4-0
     open_nemotron=nvidia/nemotron-3-ultra-550b-a55b open_beam=reflection/beam-1.0-501b-a23b"
model() { for kv in $ALL; do [ "${kv%%=*}" = "$1" ] && echo "${kv#*=}"; done; }
stage() {  # run the reasoning arms named, in parallel; Sol and Astra without the explicit-reasoning flag
  for a in "$@"; do
    case $a in frontier_sol|frontier_astra) r "$a" "$(model "$a")" & ;; *) r "$a" "$(model "$a")" $ON & ;; esac
  done
  wait
}
# The order is by expected cost only (every arm runs either way): if the spend stop is reached, it is reached in
# the last, most expensive arms, and a later run resumes them.
echo "[$(date +%H:%M)] no-reasoning arms, all models"
for kv in $ALL; do d "${kv%%=*}" "${kv#*=}" & done; wait
echo "[$(date +%H:%M)] reasoning arms, stage 1"; stage frontier_sol frontier_astra open_beam open_mistral open_nemotron open_glm frontier_gemini frontier_grok
echo "[$(date +%H:%M)] reasoning arms, stage 2"; stage open_inkling1 open_muse open_deepseek open_qwen
# Not run, to keep the cost down (decided before either had produced a reasoning answer): the reasoning arms of
# the two most expensive models, Kimi K3.1 Pro and Claude Fable 5.2. Their no-reasoning arms are run above.
#   stage open_kimi frontier_fable
echo "[$(date +%H:%M)] done"
tail -q -n 2 runs/openrouter_logs/*.log | cut -c1-300
