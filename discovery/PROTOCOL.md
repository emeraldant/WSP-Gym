# DiscoveryAgent protocol, run through Claude Code

This campaign follows the DiscoveryAgent loop (Queen et al. 2026, github.com/owencqueen/DiscoveryAgent-code)
with its prompts, but without its Python orchestrator, which needs an Anthropic API key:

| DiscoveryAgent component | Here |
|---|---|
| Director (`director_workflow.py`, `hyp_generation.py` prompts) | the main Claude Code session |
| Execution agent (Claude Agent SDK, sandboxed workspace) | Claude Code subagents, one per query, writing under `discovery/` |
| Review designer + reviewer (`review_agent.py`) | a Claude Code reviewer subagent with a task-specific prompt |
| Empirical memory (`results_memory.py`) | `checkpoints/round_N/memory_report.md`, written with CONTINUATION_REPORT_PROMPT |
| Hypothesis brancher | rounds = branches of hypotheses run in parallel subagents |
| Provenance (checkpoints + git) | every query, report and memory file committed on this branch |

Deviations: queries within a round run in parallel rather than strictly sequentially; Perplexity deep
research is off; no Census/Open Targets agents. Mode: automatic (DA-Auto), 1 exploration round then
hypothesis rounds.
