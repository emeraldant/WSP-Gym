"""Run weightgym.tinker_train on the free Inkling 2 Mini variant.

tinker_train looks up a list price for --model; the free variant has none, so register it at $0 here
(the recorded usd_train is then 0, which is what the account is charged if the offer holds).
    PYTHONPATH=. python discovery/experiments/scripts/R4_train.py sft --log-path ... --lr ... --steps ...
"""
import sys
from weightgym import tinker_train as tt

FREE = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
tt.PRICE[FREE] = {"prefill": 0.0, "cached": 0.0, "sample": 0.0, "train": 0.0}
if "--model" not in sys.argv:
    sys.argv += ["--model", FREE]
tt.main()
