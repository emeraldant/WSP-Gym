import json, os, sys, numpy as np
sys.dont_write_bytecode = True
TASK = "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness"
sys.path.insert(0, TASK); sys.path.insert(0, "/home/user/wsp-gym/discovery/experiments/scripts")
import core, generate
import R4F_audit_parsers as AP
def fwd(w, x):
    h = np.maximum(0, np.array(w["W1"]) @ x + np.array(w["b1"])); h = np.maximum(0, np.array(w["W2"]) @ h + np.array(w["b2"])); return float((np.array(w["W3"]) @ h + np.array(w["b3"]))[0])
n = eq = wit = 0; three_dec = True
for split in ("val", "test", "val_decoy", "test_decoy"):
    for it in core._frozen(split):
        n += 1
        w = AP.my_parse_weights(generate.instruction(it))
        eq += all(np.array_equal(np.array(w[k]), np.array(it["weights"][k])) for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
        three_dec &= all(abs(v * 1000 - round(v * 1000)) < 1e-6 for k in ("W1", "W2", "W3") for row in it["weights"][k] for v in row)
        if it["poisoned"]:
            x = np.array([int(c) for c in it["oracle_witness"]], float)
            intrig = all(int(x[i]) == v for i, v in it["trigger"])
            lab = int(fwd(w, x) > 0.5); sp = int(core.spec_eval(it, x[None])[0])
            wit += (intrig and lab != sp)
print("networks", n, "weights parsed back equal", eq, "all weights 3-decimal", three_dec, "witnesses in trigger and disagreeing", wit)
