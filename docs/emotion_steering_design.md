# Emotion steering by reading weights: experiment design

Status: design, nothing run. Thresholds marked *(proposed)* should be fixed in a config file before any edit
is scored, as was done for the Qwen slice.

## Question

Can a model that reads one MLP layer of Qwen3.5-0.8B as a small table propose an edit of at most 3
operations that makes the model's replies express a target emotion, without hurting what it says or how well
it reasons? And does it do better than (a) random edits to the same table, (b) a heuristic that reads the
table without any learning, and how close does it get to (c) ordinary activation steering?

The Qwen slice result (README, fourth result) shapes this design: the proposers matched the published edit
but did not beat random edits, mostly because the six-row table made a good edit easy to hit by chance. Every
choice below that differs from the Qwen slice is meant to prevent that from happening again, or to detect
it early.

## Hypotheses (fixed in advance)

| | Hypothesis | Test |
|---|---|---|
| **H1 (primary)** | Proposers beat random valid edits on the same table | Mean gain over items of each arm's best edit, minus random; 95% interval from resampling items |
| H2 | Proposers beat the table-reading heuristic | Same statistic, minus the heuristic |
| H3 | The best weight edit reaches at least half of activation steering's shift at matched side effects | Emotion shift at the same capability gate, on the test prompts |
| H4 | Edits generalise to prompt styles not used for screening | Shift on held-out prompt domains vs. on screen domains |

H1 is the one comparison the headline claim rests on. H2 to H4 are reported whichever way they come out.

## Model and layers

- **Model:** Qwen3.5-0.8B (24 layers, d_model 1024, d_mlp 3,584), the same weights as the Qwen slice, chat
  template with **thinking off** (replies are short conversational text, not reasoning traces).
- **Layers:** 4, 6, ..., 20 (9 layers). Emotion is expected in the middle layers, unlike the Wait edit, which
  worked in layers 22–23. The sweep over layers is part of the result, not tuning.
- **Edited matrix:** `mlp.down_proj` of the layer, as before, so `qwen_bench.write_delta_adapter` and
  vLLM's LoRA serving are reused unchanged.

## Phase 0: emotion directions (build, then validate)

**Rows of the table are directions in the residual stream, not tokens.**

1. **Emotions:** the six non-neutral labels of the scoring classifier: joy, sadness, anger, fear, disgust,
   surprise. Two non-emotion control directions are added: *formality* (formal vs. casual) and *length*
   (long vs. short replies). That gives 8 rows. **Targets** (the emotion an item asks for): joy, sadness,
   anger, fear. Disgust, surprise and the two control rows are distractors and also measure specificity.
2. **Contrastive data:** about 400 neutral user prompts from a *direction-building pool*, kept apart from every
   prompt used for scoring. For each prompt and each emotion, generate one reply with a system prompt asking
   for that emotion ("Reply as someone who feels deeply joyful") and one with a neutral system prompt.
3. **Direction:** at the output of layer L (the residual stream after layer L, where `down_proj` writes),
   take the mean activation over the *reply* tokens of each reply, with the system prompt removed before the
   forward pass, so that the direction comes from the text rather than the instruction. d_e(L) = mean(emotional)
   − mean(neutral), normalised to unit length. 9 layers × 8 directions.
4. **Validation, on held-out prompts:**
   - a linear probe on d_e separates emotional from neutral replies with AUC ≥ 0.9 *(proposed)*;
   - **activation steering** (adding α·d_e at layer L at every token, α tuned on the screen prompts) shifts the
     classifier toward the target emotion by ≥ 0.3 in probability *(proposed)*, with capability loss under the
     gate defined below.

**Gate 0.** A (layer, target) pair whose direction fails validation is dropped before any table is built. If
activation steering cannot move an emotion at any layer, a weight edit in that layer has no reason to work, and
that emotion is dropped.

## Phase 1: the table

For layer L, D is the 8 × 1024 matrix of directions and W the 1024 × 3,584 `down_proj` matrix.

- **Printed matrix:** P = D W (8 × 3,584). P[e, j] = how strongly neuron j writes emotion e into the residual
  stream each time it fires. Read straight from the weights, with no activations and no forward passes, the
  same idea as the Wait slice, except that the rows are directions instead of unembedding rows.
- **Shown columns:** the 64 columns with the largest variance across the 8 rows, chosen **independently of the
  target**, so the table for a layer is the same for all four targets. A proposer that ignores the requested
  emotion gives the same answer for all four, and that can be checked.
- **Units:** the shown entries are scaled to standard deviation 1, as before.
- **Edit to real-weight mapping:** exactly `SliceItem.real_delta` / `factors` from `qwen_slice.py`, with D in
  place of U: column operations act on neurons' whole output columns, and row operations and single-weight sets
  use the minimum-norm lift G = Dᵀ(DDᵀ)⁻¹. At most 3 operations, rank ≤ 8, served as one LoRA adapter.

### Two variants of the edit format (both fixed in advance)

The Qwen slice's main weakness was that whole-row edits made a good edit easy to hit by chance. Here, scaling
the target row (for example ×2 on joy) amplifies every neuron's existing writes along d_joy, which is close to
activation steering in one move, and a random edit hits it often.

- **Full format:** all operations allowed, the same as the Qwen slice.
- **Column-only format (primary for H1):** row operations are invalid, so the proposer has to pick neurons.
  This is the variant in which reading the weights can matter.

### Gate 1: is the table too easy?

Before any proposer is run, score 64 random valid edits per (layer, target) in each format. If more than 25% of
random edits *help* (definition below) *(proposed)*, the table is too easy and has to be made harder (more
shown columns, more distractor rows, or a smaller budget) and Gate 1 is run again, still before any proposer is
scored. The Qwen slice would have failed this gate: 20 of 61 random whole-row edits helped.

## Phase 2: proposers and arms

Each item is one (layer, target): 9 × 4 = 36 items. Each arm gives 8 answers per item and format, from the
same prompt.

| Arm | What it is | Why |
|---|---|---|
| Unedited | No edit | Baseline and noise floor |
| Random | Random valid edits from `baselines.random_policy` | The comparison for H1 |
| Shuffled table | Each proposer again, with the printed numbers shuffled; its answer is applied to the real weights | Shows whether the numbers informed the answer |
| Table heuristic | Scale the k shown neurons with the largest P[target, j] by a factor f (k ≤ 3, f from {1.5, 2, 3}, best on screen) | The comparison for H2: reading the table without learning |
| Untrained Inkling 2 Small, direct and with reasoning | Off the shelf | Zero-shot reference |
| Existing RL proposers (formula, real-data) | Trained on small networks only | Do skills learned in the gym carry over to steering? |
| Tone proposer (optional, Phase 3) | Trained on a new synthetic reprogramming task | See Phase 3 |
| In-class gradient search | Optimise a rank-≤3 edit on the shown columns against the scorer on the screen prompts | Upper bound for what the edit format can do |
| Activation steering | α·d_e added at run time, at layer L | Reference for H3 (not a weight edit) |
| System prompt | "Respond in a very joyful tone", no weight change | Cheapest baseline, as the Wait token ban was |

**Proposer prompt** (adapted from `SliceItem.prompt`): what the layer is, what the rows mean (one sentence per
emotion or control), the current behaviour on the screen prompts (mean classifier probability of each
emotion), the target ("make the model's replies express more **joy**, without changing what they say or
making them worse"), the shown weights, and the edit format. The prompt for the other targets differs only in
the target word.

## Scoring an edited model

### Prompts (disjoint from the direction-building pool)

| Set | Size | Use |
|---|---|---|
| Screen | 64 | Scoring every distinct edit |
| Dev | 256 | Best edit per arm and item |
| Test | 500 | Final runs; never seen before the last phase |
| Test, held-out styles | 200 | H4: domains absent from screen and dev (for example multi-turn, factual Q&A) |

The prompts are neutral and open-ended: advice, descriptions, short stories, explanations. Decoding: thinking
off, temperature 0.7, top-p 0.95, at most 256 new tokens, 1 sample per prompt (4 seeds for the unedited
model, to measure noise).

### Measures

1. **Emotion shift (main measure):** the mean probability of the target label from a fixed off-the-shelf
   7-label emotion classifier, edited minus unedited. Second scorer: a separate, fixed LLM judge that rates
   the target emotion's intensity from 1 to 5 on 100 replies per condition. Classifier–judge agreement is
   reported. If the two disagree on the ranking of the top edits, the result is reported as unresolved.
2. **Specificity:** the shift on the other five emotions. An edit that raises joy and anger alike is
   reported as arousal, not joy.
3. **Lexical-only check:** the rate of emotion words from a fixed lexicon. If the classifier shift is
   explained entirely by emotion words, the edit is reported as lexical.
4. **Content kept:** the judge answers "does this reply still answer the prompt?" (yes/no). Must stay ≥ 90%
   of the unedited rate *(proposed)*.
5. **Capability gate:** GSM8K accuracy (thinking off, 128 test problems) within 2 points of unedited, and
   mean next-token KL from the unedited model on 200 fixed WikiText passages under a fixed bound
   *(proposed)*. The gate is applied only to edits that pass screening, to keep it cheap.

**Helps** = emotion shift on the screen prompts above the unedited model's seed noise (mean + 2 SD over 4
seeds), with the content check passed. This is the same rule as the Qwen slice.

## Comparison fixed in advance (H1)

For each arm, format and item: every distinct valid edit is scored on screen; the best edit that helps goes to
dev, and the capability gate is applied there. The statistic is the dev emotion shift of each arm's best gated
edit, averaged over the 36 items (zero where no edit helped or passed the gate), minus the same for random
edits. 95% intervals come from resampling items, with layers and targets resampled as one block per layer, so
the nine layers are the unit of variation. Counts of single answers that helped are reported beside it, and the
shuffled-table control beside each proposer.

**Primary:** column-only format, best RL proposer minus random. Everything else is secondary.

## Phase 3 (optional): train a proposer for this

Only if Phase 2 shows that the existing proposers carry over at all (any arm above random on the shuffled-table
control):

- **Synthetic task in the gym:** a new reprogram kind in `weightgym/reprogram.py`. The network has a main
  output and a "tone" output. The task asks for a shift of the tone output toward a target while keeping the
  main-task accuracy (recovery ≥ 0.95, the existing threshold). Exact checks, cheap generation, then RL with
  the existing Tinker loop.
- **Or RL on the slice itself:** reward = screen emotion shift with the capability gate. Train on layers 4–12
  and targets joy and sadness; evaluate on layers 14–20 and targets anger and fear, so that the test items are
  never trained on.

## Phase 4: final runs

On the test set and the held-out styles: the top three gated edits from Phase 2/3, the table heuristic,
in-class gradient search, activation steering at matched capability cost, the system prompt, and the unedited
model. Then 50 replies per condition are read by a person, blind to the condition, to check that the numbers
describe the text.

## Cost (rough)

Screening is much cheaper than for the math slice: 64 prompts × 256 tokens ≈ 16k generated tokens per edit, vs.
up to 8,192 tokens per math problem. At most 36 items × 2 formats × about 8 arms × 8 answers ≈ 4,600 edits
(fewer after deduplication) ≈ 75M tokens on a 0.8B model, a few GPU-hours with vLLM and LoRA hot-swapping,
using the existing `scripts/qwen_serve.sh`. Building the directions takes about 400 × 9 generations and one
forward pass per reply. Proposer sampling on Tinker costs about the same as the Qwen slice.

## Code plan

| File | Contents |
|---|---|
| `weightgym/emotion_dirs.py` | Contrastive generation, directions per layer, probe and activation-steering validation (Gate 0) |
| `weightgym/emotion_slice.py` | `EmotionSliceItem(layer, target, format)`: table, prompt, validation (column-only), `real_delta` / `factors` generalised from `qwen_slice.py` with D in place of U; propose / screen / dev / table commands as in `qwen_slice.py` |
| `weightgym/emotion_bench.py` | Generation with an adapter, classifier and judge scoring, content check, capability gate |
| `tests/test_emotion_slice.py` | Edit → real-delta equivalence (as for the Qwen slice), column-only validation, a fixed table for a fixed seed |

The building, validation and table code can be written and tested without a GPU. Generation, directions and
scoring need the GPU environment with vLLM.

## What would count as a result

- **Positive:** in the column-only format, an RL proposer beats random edits (interval above zero), beats the
  shuffled-table control and holds its shift on the held-out styles. That would be the first evidence in this
  project that reading the weights adds something on a real LLM.
- **Null, reported as one:** proposers no better than random or the heuristic, as in the Qwen slice. This is
  still informative if Gate 1 passed, because then the table was not giving the answer away.
- **Either way:** where in the network emotion can be steered by a 3-operation edit, and how close a baked-in
  edit gets to activation steering. Both are useful regardless of how the proposers do.
