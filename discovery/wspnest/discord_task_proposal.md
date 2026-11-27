Message for the WSPNest Discord, #task-proposals (paste as is; Discord renders the markdown).

**Task proposal: backdoor-witness (find a backdoor's trigger from a small network's weights)**

Hi everyone! This is the CPU-only version of the "implant audit" idea I posted in #ideation-and-brainstorming. It's an original `localize` task, built and ready to open as a PR.

The agent gets the weights of a small ReLU network (about 800 to 2,100 weights) as text, plus one line saying what the network should compute. Half the networks carry a rare backdoor (wrong on fewer than 1 in 4 million inputs); the other half are matched clean twins. The agent answers with an input that makes the network break its spec, or says it is clean. No code, one model call.

The verifier is a forward pass, so there is no judge. All 450 val/test networks are certified by a mixed-integer check over every input, and again by a second, independently written one: a clean twin is never wrong, and a poisoned twin is wrong only inside its trigger. The reference solution scores 1.0 on all 450 tasks under Harbor.

Results on the hard split (100 networks): random witness 0.00, always "clean" 0.50, GPT-6.2 Sol 0.85 (finds 84% of triggers). Sonnet 5.5, Haiku 5.5 and Muse Spark Open find 0-2%.

Two questions before I open it:
1. Headroom. A scripted search with 100k forward passes scores 0.91. It is outside the protocol (no code allowed), but I listed it on the card anyway. Does that count against the "no baseline at 0.9" bar? If so I'll build a harder setting first.
2. Base branch. I built on `wspnest-quality` (#1) because the task uses `wspnest check` and the `selfcheck.py` hook, like #3. Should I target that, or `benchmark`?

One note: my Docker host refuses Harbor's network sidecar, so I borrowed @Xiao Cui's fix from #2 (Docker switches the network off itself). It would be good if someone ran the oracle once with Harbor's own network policy.
