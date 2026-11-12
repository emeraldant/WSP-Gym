#!/bin/bash
# Reproduce three published MATH-500 numbers for Qwen3.5-0.8B before trusting anything new from this harness:
# unedited (Beaker: 6.2-8.4% across runs), layer 22 neuron 573 x2 (37.4% / 36.6%), "Wait" token ban (37.6%).
# One sampled answer per problem here (the published runs use four); everything else follows the published protocol.
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=${WG_QWEN_PY:-$HOME/wg-qwen/vllm-env/bin/python}
N=${N:-500}
$PY -m weightgym.qwen_bench adapter x2 '[{"layer":22,"matrix":"down","neuron":573,"op":"scale","factor":2}]' > /dev/null
$PY -m weightgym.qwen_bench eval --arm unedited --set math500 --n $N > runs/qwen/repro_unedited.log 2>&1 &
$PY -m weightgym.qwen_bench eval --arm n573_x2 --adapter x2 --set math500 --n $N > runs/qwen/repro_x2.log 2>&1 &
$PY -m weightgym.qwen_bench eval --arm wait_ban --ban-wait --set math500 --n $N > runs/qwen/repro_ban.log 2>&1 &
wait
$PY -m weightgym.qwen_bench table
