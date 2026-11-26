Mirror of the WSPNest task `backdoor-witness` (an original task for github.com/atharva-naik/auto-weight-space-programming,
`wspnest/tasks/backdoor-witness/`, plus `wspnest/tests/test_backdoor_witness.py` in `tests/`). Kept here so the work is
on GitHub until it is submitted as a pull request to WSPNest.

- Start with `HEADROOM.md`: the design record, every baseline, the model runs, the standing against each quality bar
  and the limits. `task.yaml` is the task card.
- `../backdoor-witness.bundle` is the exact commit, made on top of WSPNest pull request #1 (branch `wspnest-quality`,
  commit a3b4044). To put it on a fork, inside a clone that has that branch:
  `git fetch <path>/backdoor-witness.bundle task/backdoor-witness:task/backdoor-witness`, then push the branch.
- `../PR_backdoor-witness.md` is the draft pull-request description.
- Final report: `discovery/reports/R4F.md`. Pre-registered criteria: `discovery/checkpoints/round_4/`.
- The task was audited by a separate agent after it was first finished. Its brief and report are
  `discovery/checkpoints/round_4/audit_brief.md` and `audit_report.md`; its checks are
  `discovery/experiments/scripts/R4F_audit_*.py`. A second reviewer then checked the fixes (`audit_brief_2.md`,
  `audit_report_2.md`). `R4F.md` lists what was done about each finding of both.

The test file expects the WSPNest layout (`tasks/backdoor-witness/` next to `tests/`); it runs in a WSPNest checkout,
not from this mirror.
