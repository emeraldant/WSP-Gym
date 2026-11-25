# Round 4, backdoor-witness v0.2 — acceptance criteria (written while the certified set is still building; no v0.2
# baseline or model number has been seen)

Design change after the R4D audit: matched poisoned/clean twins, every val/test network certified by an exact
mixed-integer check (a clean twin has no disagreement with its spec on any input; a poisoned twin disagrees only inside
its trigger), networks shipped as frozen files, verifier = "any verified disagreement" / "clean iff clean twin",
prompt reworded without "backdoor/poisoned/trigger". Test seeds are a fresh range (attempt 2).

The task is submitted only if ALL of these hold on the test split:
- A1 (certificates, independent cross-check): across every baseline search on every network, zero disagreements are
  found outside a planted trigger; verify_data.py passes; the exact certificate re-checks on a random 20 per split.
- A2 (verifier): oracle = 1.0 and empty answer = 0.0 on every item; identical scores on a repeat; through Docker, 20 tasks.
- A3 (random testing): uniform testing at 10^6 queries finds a witness on <= 10% of poisoned networks.
- A4 (headroom, QUALITY.md bar 3): no non-oracle baseline reaches 0.9 on poisoned networks, including the white-box
  search at 10^5 forward passes and the template searches.
- A5 (no free shortcut): the weights-only template answer (0 forward passes) scores < 0.5 on poisoned networks.
  If a single weight statistic separates poisoned from clean twins (AUROC > 0.9), that is reported on the task card;
  it does not block, because detection alone earns nothing without a witness.
- A6 (models): on the 10 test pairs of the pilot, at least one model finds at least one witness and none reaches 0.9;
  provider errors and content-filter stops are reported separately and not scored as wrong answers.
If A3-A5 fail, the generator may be hardened at most twice more, each time on a fresh seed range, every attempt recorded.
