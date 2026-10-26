---
layout: distill
published: false
title: "Beyond Pretraining Then Post-Training: An Exploration of Interleaving NTP and RL"
title_html: "Beyond Pretraining Then Post-Training:<br>An Exploration of Interleaving NTP and RL"
description: "We move RL into the middle of pretraining or SFT and ask whether it helps the next-token prediction (NTP) training that follows. Across five studies in chess and math, it does not."
date: 2026-09-12
permalink: /blog/2026/interleaving-pt-rl/
tags: pretraining reinforcement-learning research
categories: research
authors:
  - name: Leon Li
    url: https://leonlixyz.github.io/
    affiliations:
      name: New York University
      url: https://www.nyu.edu/
  - name: Jingyan Shen
    url: https://jy-evangeline.github.io/
    affiliations:
      name: New York University
      url: https://www.nyu.edu/
  - name: Pavel Izmailov
    url: https://izmailovpavel.github.io/
    affiliations:
      name: New York University
      url: https://www.nyu.edu/
toc:
  - name: "TL;DR"
  - name: "Background"
  - name: "Setup"
  - name: "Does earlier RL help the next NTP stage?"
  - name: "Does interleaving improve the final model?"
  - name: "What the results suggest"
  - name: "Related work on mixing SFT and RL"
  - name: "Open questions"
  - name: "Evaluation details and limits"
  - name: "Complete results"
  - name: "Data and results"
bibliography: interleaving-pt-rl.bib
_styles: |
  html body d-title h1 { grid-column: text-start / page-end; }
  @media (min-width: 1180px) { html body d-title h1 { font-size: 46px; } }
  @media (min-width: 1024px) and (max-width: 1179px) { html body d-title h1 { font-size: 32px; } }
  d-article .interleave-table { overflow-x: auto; max-width: 100%; margin: 1.5rem 0; }
  d-article .interleave-table table { width: 100%; font-size: 12px; line-height: 1.55; font-variant-numeric: tabular-nums; margin: 0; }
  d-article .interleave-table th, d-article .interleave-table td { padding: 8px 10px; vertical-align: top; }
  d-article .interleave-table th { white-space: nowrap; }
  d-article .interleave-table td:first-child { min-width: 150px; }
  d-article .interleave-table td:not(:first-child) { white-space: nowrap; }
  d-article .interleave-figure img { display: block; width: 100%; height: auto; background: white; }
  d-article .interleave-figure figcaption { margin-top: 0.8rem; }
  d-article details { margin-bottom: 1rem; }
  d-article details summary { cursor: pointer; font-weight: 600; }
  d-article details .interleave-table { margin-bottom: 0; }
  d-article h2 { scroll-margin-top: 80px; }
  d-article .interleave-text-table td, d-article .interleave-text-table th { white-space: normal !important; text-align: left !important; }
  d-article mjx-container[display="true"] { overflow-x: auto; overflow-y: hidden; max-width: 100%; padding: 0.7rem 0; }
  d-article .interleave-summary { border-left: 3px solid #b66b35; padding: 0.2rem 0 0.2rem 1.1rem; margin: 1.5rem 0; }
  d-article .interleave-summary ol { margin-bottom: 0; }
  d-article .interleave-note { font-size: 0.85rem; color: var(--global-text-color-light, #6c757d); margin-top: -0.6rem; }
---

Language models are usually trained in stages. Pretraining, midtraining, and supervised fine-tuning (SFT) come first. In all three, the model is trained with next-token prediction (NTP): it imitates a fixed dataset by minimizing the cross-entropy loss of each next token. Reinforcement learning (RL), predominantly GRPO-style training, comes after that. In RL, the model samples rollouts for a set of prompts, each rollout gets a reward, and the model is then updated based on each rollout's advantage, normalized within its group.

This order means RL only starts after the model has finished all of its imitation learning on a fixed dataset. People usually do not learn this way. A student does not attend every lecture of a course first and only then start the problem sets. They go back and forth between the two, and working through problems often strengthens their understanding of the material, which better prepares them for the next lecture. Does the same learning paradigm apply to language models? Here we study whether RL can be interleaved with pretraining or SFT instead of always running last. Specifically, we want to answer the following questions:

1. **Does RL help the model learn better from the NTP data that comes after it?** To test this, we split the pretraining or SFT tokens into two halves. We first train on the first half, run RL in the middle, and then train on the second half. If RL is beneficial for NTP, the model should reach a lower validation loss.
2. **Does interleaving SFT and RL improve the final performance?** Even if RL helps with the subsequent pretraining, the compute that RL uses does not come for free. So we compare whole training pipelines under the same budget: we split the RL budget in half, run the first half in the middle and the second half after NTP finishes, and compare against a baseline that spends the entire RL budget in the final stage.

We run five studies on chess and math, with model sizes ranging from a 47M chess model to Llama 3.2 3B. The answer to the first question is no, and the answer to the second is not consistently.

<h2 id="tl-dr">TL;DR</h2>

<div class="interleave-summary" markdown="1">

**Does earlier RL help next-token prediction?** Across five studies on chess and math, we find no evidence that it does, and no consistent improvement in the final model.

1. **Continuing NTP after RL leads to catastrophic forgetting.** In chess, RL improves pass@1 from 13.9% to 30.8%, but after the second half of pretraining it drops back to 17.9%, about the same as a model that never saw RL (18.2%). The validation loss is also about the same, so RL does not help the model learn from the remaining data. Chess SFT, math pretraining, and Llama SFT show the same pattern.
2. **Mixing RL traces into the NTP data prevents forgetting, but does not help NTP either.** We collect the correct rollouts from RL (traces) and mix them into the remaining pretraining or SFT data. The model keeps most of the RL gain, even if we restart from the checkpoint before RL. However, the validation loss stays the same or gets slightly worse. The model learns to reproduce RL's answers, not to learn the original data better.
3. **The advantage disappears after the final RL.** The trace runs start the final RL far ahead, but once every schedule has used the same RL budget, the baseline matches or beats them in four of five studies. The only exception is a 0.6-point gain on Llama's RL task.
4. **Two results are mixed.** In Llama, every interleaved schedule does better on MATH-500 and worse on GSM8K. Separately, RL → trace training → RL after pretraining finishes sometimes beats the baseline. But there is no NTP data left at that point, so it does not answer our question, and we only report it as a reference.
5. **Published methods that gain from mixing SFT and RL add something we did not.** They train on demonstrations from a much stronger teacher, and they choose which questions receive supervision, for example the questions on which every rollout fails. In their own ablations, alternating SFT and RL without that choice does not beat SFT followed by RL, which matches our results.

</div>

<figure class="l-page interleave-figure"><a href="/assets/blog/interleaving-pt-rl/head-start-summary.svg" target="_blank" aria-label="Open the summary figure at full size"><img src="/assets/blog/interleaving-pt-rl/head-start-summary.svg" loading="lazy" alt="Five panels, one per study. Each shows task accuracy before and after final RL for the baseline, direct continuation, and trace training. Trace runs start final RL well above the baseline, but the baseline catches up or passes them in every panel except Llama, where traces finish 0.6 points ahead."></a><figcaption>Figure 1. Task accuracy at the end of supervised training ("before final RL") and after all RL updates ("after final RL"). The baseline does all 3,000 RL updates at the end. The other schedules do 1,500 RL updates between the two halves of supervised training and 1,500 at the end. Llama shows greedy accuracy on held-out Polaris, its RL task; the other panels show sampled pass@1. The 64×16 chess study has no direct-continuation run. Lines connect two evaluations and are not learning curves.</figcaption></figure>

<h2 id="background">Background</h2>

This work builds on our previous paper, [Understanding Reasoning from Pretraining to Post-Training](https://arxiv.org/abs/2607.16097), where we studied how pretraining affects the RL that follows. In both chess and math, a lower pretraining loss predicts a higher accuracy after RL, and models pretrained on more tokens improve faster with the same RL compute (Figures 2 and 3).

<figure class="l-page interleave-figure"><a href="/assets/blog/interleaving-pt-rl/prior-work-chess.svg" target="_blank" aria-label="Open the prior-work chess figure at full size"><img src="/assets/blog/interleaving-pt-rl/prior-work-chess.svg" loading="lazy" alt="Earlier chess study: lower pretraining loss predicts higher fitted post-RL reward; more pretraining tokens predict a steeper RL improvement slope; adding model size improves the slope prediction."></a><figcaption>Figure 2. Chess results from the earlier paper. Left: fitted reward at a fixed reference RL compute, R<sup>ref</sup>, against pretraining loss. Each color is one RL compute level C, and dashed lines are linear fits. Middle: the fitted gain per tenfold increase in RL compute, B, against pretraining tokens T. Right: predicting B from both T and model size N. Reproduced from <a href="https://arxiv.org/abs/2607.16097">Shen et al.</a></figcaption></figure>

<figure class="l-page interleave-figure"><a href="/assets/blog/interleaving-pt-rl/prior-work-math.svg" target="_blank" aria-label="Open the prior-work math figure at full size"><img src="/assets/blog/interleaving-pt-rl/prior-work-math.svg" loading="lazy" alt="Earlier OLMo2 math study: pretraining loss predicts fitted post-RL pass at 1, pretraining tokens predict the RL improvement slope, and pretraining loss also predicts pre-RL performance."></a><figcaption>Figure 3. Math results from the earlier paper, using OLMo2 1B checkpoints pretrained on 10B–200B tokens. Left and middle: the same two relationships as in chess. Right: pass@1 after SFT and before RL, R<sub>0</sub>, against pretraining loss. Reproduced from <a href="https://arxiv.org/abs/2607.16097">Shen et al.</a></figcaption></figure>

These results are about one direction: a better pretrained model benefits more from RL. Here we ask about the other direction: does RL help the model learn better from the pretraining or SFT data that comes after it?

Either outcome is plausible. On one hand, a model can already solve some tasks long before pretraining ends, and practicing these tasks with RL might teach it skills that make the remaining data easier to learn. On the other hand, NTP and RL optimize for different things. NTP pushes the model toward the exact tokens in a fixed dataset, while RL pushes it toward whatever rollouts get a high reward, which are often not in the dataset. So when training switches back to NTP, it can pull the model away from what it learned in RL, which is catastrophic forgetting.

Several recent training pipelines already alternate between NTP and RL, but for different reasons. MAI-Thinking-1 trains midtrained checkpoints on RL rollouts and then resumes RL, to recover from unstable RL runs and to transfer progress to newer base checkpoints ([report, §3.1.4](https://microsoft.ai/pdf/mai-thinking-1.pdf#page=34)). Ring-Zero selects and shortens correct RL trajectories, fine-tunes the original base model on them, and then continues RL ([paper, §3.1.2](https://arxiv.org/html/2607.12395v2#S3.SS1.SSS2)). ReMiT uses an RL-tuned model to reweight the token losses during midtraining ([paper, §3](https://arxiv.org/html/2602.03075v1#S3)). We test two simpler variants: continuing NTP directly from the RL checkpoint, and mixing the correct RL rollouts into the NTP data.

<h2 id="setup">Setup</h2>

In each study, we split the pretraining or SFT data into two halves, NTP A and NTP B. In the pretraining studies, each half is pretraining text mixed with some task data. In the SFT studies, each half is a set of worked solutions, and all runs start from the same pretrained checkpoint. We compare three schedules (Figure 4). The baseline trains on both halves and then runs all 3,000 RL updates. Direct continuation runs 1,500 RL updates between the two halves and starts NTP B from the RL checkpoint. The trace schedule also runs 1,500 RL updates in the middle, and mixes the correct rollouts of that stage into NTP B.

<figure class="l-body interleave-figure"><picture><source media="(max-width: 600px)" srcset="/assets/blog/interleaving-pt-rl/schedules-compact.svg"><img src="/assets/blog/interleaving-pt-rl/schedules.svg" loading="lazy" alt="Three schedules on one axis. Baseline: NTP A, then NTP B, then 3,000 RL updates. Direct continuation: NTP A, then 1,500 RL updates, then NTP B starting from the RL checkpoint, then 1,500 RL updates. Traces: NTP A, then 1,500 RL updates, then NTP B with the correct rollouts of that RL stage mixed in, then 1,500 RL updates."></picture><figcaption>Figure 4. The three schedules compared in every study. All three train on the same NTP data and use 3,000 RL updates in total. They differ only in where the first 1,500 RL updates go and in what NTP B receives from them. Block widths show order and RL update counts, not compute.</figcaption></figure>

A **trace** is a rollout from RL that gets the correct answer. In the trace schedule, we collect the traces from the first RL stage and mix them into NTP B, so the model trains on both the remaining data and its own correct rollouts. NTP B can either start from the RL checkpoint, or restart from the checkpoint before RL (the end of NTP A). Restarting tests whether the traces alone are enough to carry over what RL learned, without keeping any of the RL weights. We specify which one each run uses.

Some runs also split the RL prompts. **A/B** means the two RL stages use two disjoint halves of the prompts, and **full/full** means both stages use all prompts.

Within each study, all schedules train on the same NTP data and use the same RL budget of 3,000 updates. The baseline spends all 3,000 at the end, while the interleaved schedules spend 1,500 in the middle and 1,500 at the end. So at the end of NTP B, the interleaved models have already had 1,500 RL updates and the baseline has had none, and comparisons at this point favor the interleaved models. The trace runs also train on extra tokens, and some runs reset the optimizer between stages while others do not. So we control the data and the number of RL updates, but not the exact total FLOPs.

<div class="interleave-table interleave-text-table" markdown="1">

| Study | Model | Supervised data | RL task · prompts × samples per update |
| --- | --- | --- | --- |
| Chess PT, 256 × 8 | 47M | 5B pretraining tokens + chess SFT | Chess puzzles · 256 × 8 |
| Chess PT, 64 × 16 | 47M | 5B pretraining tokens + chess SFT | Chess puzzles · 64 × 16 |
| Math PT | OLMo2 1B | 45B pretraining tokens + chain-of-thought data | SkyEasy math · 64 × 16 |
| Chess SFT | 47M, from a fixed 9.18B-token pretrained model | 77,717 SFT examples | Chess puzzles · 64 × 16 |
| Llama SFT | Llama 3.2 3B | 823,505 NuminaMath-CoT examples | Polaris math · 64 × 16 |

</div>

We measure the two questions at different points:

1. **Does RL help the model learn better from the NTP data that comes after it?** We evaluate every model at the end of NTP B. If RL helps, the interleaved models should have a lower validation loss than the baseline. This should also hold for runs without traces, since traces add new data and we want to know whether RL helps with the original data.
2. **Does interleaving SFT and RL improve the final performance?** We evaluate every model again after the final RL, when all schedules have used the same RL budget.

**Validation loss** is the cross-entropy on held-out data from the same source as the training data; lower is better. For chess puzzles and SkyEasy, we sample 16 rollouts per problem and report pass@k: **pass@1** is the probability that a single rollout is correct, and **pass@16** is the probability that at least one of the 16 is correct. For the Llama benchmarks, we report greedy accuracy.

<h2 id="does-earlier-rl-help-the-next-ntp-stage">Does earlier RL help the next NTP stage?</h2>

<h3>Continuing NTP after RL: catastrophic forgetting</h3>

The most direct way to test this is to run RL in the middle of pretraining and then continue pretraining from the RL checkpoint. We do this in the 256 × 8 chess study: we pretrain on 2.5B tokens, run 1,500 RL updates, and then pretrain on the remaining 2.5B tokens from the RL checkpoint with a fresh AdamW optimizer. We compare against two runs without RL: one that pretrains on all 5B tokens in a single run, and one that pretrains on the same two 2.5B halves in two separate runs.

<div class="interleave-table l-page" markdown="1">

| Checkpoint | PT loss ↓ | Pass@1 (%) | Pass@2 (%) | Pass@4 (%) | Pass@8 (%) | Pass@16 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| After the first 2.5B tokens | 0.525055 | 13.85 | 22.14 | 32.05 | 42.07 | 51.01 |
| After first RL | 0.540289 | 30.79 | 36.39 | 41.26 | 45.60 | 49.53 |
| After direct continuation | 0.507826 | 17.85 | 27.68 | 38.27 | 47.86 | 55.68 |
| Two 2.5B schedules, no RL | 0.507432 | 18.23 | 28.09 | 38.71 | 48.62 | 57.43 |
| One 5B schedule, no RL | 0.512429 | 19.11 | 29.06 | 39.58 | 49.21 | 57.30 |

</div>

<figure class="l-body interleave-figure"><img src="/assets/blog/interleaving-pt-rl/direct-continuation.svg" loading="lazy" alt="Pass at k after the first RL stage, after direct PT continuation, and after uninterrupted PT."><figcaption>Figure 5. Pass@k after first RL, after 2.5B more pretraining tokens from the RL weights, and after uninterrupted pretraining. After continued pretraining, the RL model's curve is close to that of a model that never had RL.</figcaption></figure>

RL more than doubles pass@1, from 13.85% to 30.79%. However, after the second half of pretraining, pass@1 drops back to **17.85%**, slightly below the 18.23% of the split run without RL. The validation loss also ends at almost the same value: 0.507826 with RL in the middle and 0.507432 without. By both measures, the final model looks as if RL never happened: the second half of pretraining erases what RL learned, and RL does not help the model learn the pretraining data.

The same thing happens when the RL gain comes from traces. In another run, we train on RL traces after the first 2.5B tokens instead of running RL directly, which raises pass@1 to 30.27%. After continuing with plain pretraining, pass@1 falls back to **17.72%**.

The other studies show the same pattern:

<div class="interleave-table" markdown="1">

| Study | After first RL | After NTP B | Baseline |
| --- | ---: | ---: | ---: |
| Chess PT, 256 × 8 | 30.79 | 17.85 | 18.23* |
| Math PT, fresh optimizer | 16.76 | 10.25 | 9.28 |
| Math PT, optimizer state kept | 16.44 | 10.39 | 9.28 |
| Chess SFT, full/full | 25.50 | 11.60 | 11.36 |
| Llama SFT, full/full | 13.96 | 9.38 | 7.62 |

</div>

<p class="interleave-note">In every row, NTP B continues from the RL checkpoint. Values are pass@1 (%): puzzle accuracy for chess, SkyEasy accuracy for math, and greedy accuracy on held-out Polaris for Llama. "Baseline" is the baseline at the end of NTP B. *For chess, this is the split run without RL; math has no split run, so we use the single 45B-token run.</p>

In math, carrying over the optimizer state from RL instead of resetting it makes no difference. Math and Llama end NTP B one to two points above their baselines, so a small part of the RL gain remains, while the chess studies end level with theirs.

The validation loss tells the same story. In chess pretraining, no run with RL in the middle ends NTP B with a meaningfully lower loss than the split run without RL: all are higher or within 0.0001. In math, all four interleaved runs end with a slightly higher loss than the baseline, 1.2347–1.2359 vs. 1.2276 (the fourth is a trace run whose final RL results are not available yet; see the full tables below). The only exception is Llama, where the two direct-continuation runs have a slightly lower loss than the baseline, 0.4441 and 0.4440 vs. 0.4455. However, these runs reset the optimizer between stages, while the baseline trains SFT A and SFT B with a single optimizer and learning-rate schedule, so we do not treat a 0.0015 difference as evidence that RL helps.

Continuing after RL does change one thing. In the chess table, RL improves pass@1 but slightly lowers pass@16, from 51.01% to 49.53%: the model gets better at answering correctly on the first try, but slightly worse at finding a correct answer within 16 tries. The second half of pretraining brings pass@16 back up to 55.68%, close to the runs without RL, but pass@1 drops at the same time.

<h3>Mixing RL traces into the NTP data</h3>

A standard way to mitigate catastrophic forgetting is replay: keeping some data from the earlier task in later training. Here, this means mixing the traces from the first RL stage into NTP B. In several runs, we also restart NTP B from the checkpoint before RL, so the traces are the only thing carried over from RL.

<div class="interleave-table" markdown="1">

| Study | Starts from | First RL | Baseline | Traces |
| --- | --- | ---: | ---: | ---: |
| Chess PT, 256 × 8 | NTP A | 30.79 | 18.23* | 33.50 |
| Chess PT, 256 × 8 | RL | 30.79 | 18.23* | 34.04 |
| Chess PT, 64 × 16 | RL | 29.17 | 19.11 | 30.24 |
| Math PT | NTP A | 16.76 | 9.28 | 16.83 |
| Chess SFT | NTP A | 25.50 | 11.36 | 26.74 |
| Llama SFT | NTP A | 14.45 | 7.62 | 15.92 |

</div>

<p class="interleave-note">"Starts from" is the checkpoint NTP B starts from: before RL (end of NTP A) or after RL. "First RL" is the accuracy of the RL checkpoint that generated the traces. "Baseline" and "Traces" are measured at the end of NTP B. Same metrics as the previous table. Chess 64 × 16 and chess SFT use full/full prompts; Llama uses A/B. *The split chess run without RL.</p>

With traces, the model keeps the RL gain, even when NTP B restarts from the checkpoint before RL. In every row, the trace run ends NTP B above the RL checkpoint that generated its traces. For example, in chess SFT, restarting from SFT A and training on SFT B plus traces reaches 26.74% pass@1, higher than the 25.50% of the RL checkpoint itself.

However, the validation loss does not improve. Against the same baselines, the trace runs fit the original data no better, and usually slightly worse:

- **Chess PT:** 0.5110 and 0.5105, against 0.5074 for the split run without RL.
- **Math PT:** 1.2348, against 1.2276.
- **Chess SFT:** held-out pretraining loss of 0.7966, against 0.7222. This study measures loss on pretraining text, not on SFT solutions.
- **Llama SFT:** Numina loss of 0.4534 and 0.4520, against 0.4455.

So traces make the model better at the RL task without making it better at the data it is supposed to learn. The model learns to reproduce RL's correct answers, which prevents forgetting, but this is different from RL helping the model learn the original data, which is what our first question asks.

<details class="l-page" markdown="1"><summary>Full tables at the end of NTP B for the math PT, chess SFT, and Llama studies</summary>

<div class="interleave-table l-page" markdown="1">

| OLMo2 1B · schedule | PT loss ↓ | Pass@1 (%) | Pass@2 (%) | Pass@4 (%) | Pass@8 (%) | Pass@16 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1.227608 | 9.28 | 15.08 | 22.82 | 32.05 | 42.20 |
| Direct continuation, fresh AdamW | 1.235861 | 10.25 | 16.50 | 24.45 | 33.52 | 43.20 |
| Direct continuation, AdamW state kept | 1.235521 | 10.39 | 16.53 | 24.55 | 34.22 | 45.20 |
| PT + traces · full/full | 1.234848 | 16.83 | 24.99 | 34.09 | 43.44 | 52.60 |
| PT + traces · A/B | 1.234661 | 17.79 | 25.80 | 34.93 | 44.60 | 54.20 |

</div>

<div class="interleave-table" markdown="1">

| Chess SFT · schedule | Held-out PT loss ↓ | Pass@1 (%) | Pass@4 (%) | Pass@16 (%) |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.7222 | 11.36 | 28.64 | 48.18 |
| Direct continuation · A/B | 0.7284 | 11.66 | 29.64 | 49.73 |
| SFT + traces · A/B | 0.8167 | 22.29 | 35.35 | 46.82 |
| Direct continuation · full/full | 0.7277 | 11.60 | 29.24 | 50.14 |
| SFT + traces · full/full | 0.7966 | 26.74 | 41.93 | 55.34 |

</div>

<div class="interleave-table" markdown="1">

| Llama 3.2 3B · schedule | Numina SFT loss ↓ | GSM8K (%) | MATH-500 (%) | Polaris (%) |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.4455 | 70.58 | 39.00 | 7.62 |
| Direct continuation · A/B | 0.4441 | 69.52 | 38.80 | 9.86 |
| SFT + traces · A/B | 0.4534 | 71.49 | 39.60 | 15.92 |
| Direct continuation · full/full | 0.4440 | 70.43 | 39.40 | 9.38 |
| SFT + traces · full/full | 0.4520 | 69.60 | 38.40 | 15.23 |

</div>

</details>

**Answer to the first question: no.** When NTP continues from the RL checkpoint, the model forgets the RL gain, and the validation loss ends about where it would without RL. When we mix in RL traces, the model keeps the RL gain, but the validation loss on the original data is no lower. The gain comes from the traces, not from learning the original data better.

<h2 id="does-interleaving-improve-the-final-model">Does interleaving improve the final model?</h2>

The trace runs start the final RL far ahead of the baseline. The question is whether they are still ahead once both schedules have used their full RL budget.

<div class="interleave-table" markdown="1">

| Study | Baseline | Traces | Final difference |
| --- | ---: | ---: | ---: |
| Chess PT, 256 × 8 | 19.1 → 37.9 | 33.5 → 37.1 | −0.72 |
| Chess PT, 64 × 16 | 19.1 → 38.2 | 30.2 → 36.1 | −2.10 |
| Math PT (SkyEasy) | 9.3 → 28.9 | 16.8 → 24.9 | −4.05 |
| Chess SFT | 11.4 → 33.5 | 26.7 → 33.0 | −0.57 |
| Llama SFT (Polaris)* | 7.6 → 16.7 | 15.9 → 17.3 | +0.59 |

</div>

<p class="interleave-note">Pass@1 (%) at the end of NTP B → after the final RL. *Greedy accuracy. The trace runs are the ones in Figure 1; for chess PT 256 × 8, it is the run that restarts from the NTP A checkpoint.</p>

During its 3,000 final RL updates, the baseline improves by 19–22 points of pass@1 in the four sampled studies, and by 9 points of Polaris accuracy in Llama. The trace runs start higher but only improve by 3.6–8.1 points in their last 1,500 updates, and by 1.4 points in Llama. In the end, they finish below the baseline in four studies and 0.59 points above it in Llama. Interleaving changes when the model reaches its final accuracy, but not how high that accuracy is.

<h3>Chess pretraining</h3>

The 256 × 8 study has the most schedules. Six of them interleave RL with pretraining: direct continuation; direct continuation with traces replayed late in NTP B; two variants of pretraining plus traces with different data orders; and two runs that train on traces and then continue with plain pretraining. All six use the same 28,419 prompts for the final RL as the baseline, and they finish between **36.36% and 37.14% pass@1**, compared with **37.86%** for the baseline.

To check whether these differences are beyond noise, we run a paired bootstrap with 20,000 resamples over puzzles and adjust the intervals for multiple comparisons. Every pass@1 and pass@16 interval includes zero. So we cannot say interleaving hurts, but none of these runs finishes ahead of the baseline.

In the 64 × 16 study, both runs that mix traces into the second half of pretraining finish below the baseline: by 2.04 points for A/B (interval [−3.78, −0.28]) and 2.10 points for full/full ([−3.78, −0.39]). These are the only chess pretraining comparisons whose intervals exclude zero, and both go against interleaving. The full/full run does end with a lower pretraining loss than the baseline, 0.520700 vs. 0.534776, but this does not translate into higher accuracy.

<figure class="l-body interleave-figure"><img src="/assets/blog/interleaving-pt-rl/later-chess-final.svg" loading="lazy" alt="Final PT loss, pass at 1, and pass at 16 in the five later chess PT experiments."><figcaption>Figure 6. Final results in the 64 × 16 chess study. PT plus traces gives the lowest final PT loss and the lowest pass@1. The two trace-only runs are discussed below. Dashed lines show the baseline.</figcaption></figure>

<h3>Math pretraining</h3>

All three interleaved schedules with final results finish below the baseline on every accuracy metric: SkyEasy pass@1 is 24.36–24.89% vs. 28.94%, GSM8K is 31.31–35.63% vs. 42.53%, and MATH is 28.52–28.98% vs. 31.34%. The trace run starts the final RL 7.6 points ahead on SkyEasy and finishes 4.1 points behind.

<h3>Chess SFT</h3>

The baseline finishes with the highest pass@1, 33.54%. The two trace runs finish 0.57 and 0.88 points below it, and the two direct-continuation runs 1.91 and 2.43 points below. Direct continuation with full/full prompts has the highest pass@16, 57.16% vs. 56.01%. These are single runs without uncertainty estimates.

<figure class="l-body interleave-figure"><img src="/assets/blog/interleaving-pt-rl/chess-sft-stages.svg" loading="lazy" alt="Chess pass at 1 and pass at 16 before and after final RL. Large intermediate trace gains do not become a final pass at 1 advantage."><figcaption>Figure 7. Chess SFT, all five schedules. The trace runs enter final RL far above the baseline and finish just below it.</figcaption></figure>

<h3>Llama SFT</h3>

Llama is the only study where interleaving changes the final model by more than a point on several benchmarks, but whether it helps depends on the benchmark.

<div class="interleave-table l-page" markdown="1">

| Schedule | Numina SFT loss ↓ | GSM8K (%) | MATH-500 (%) | Polaris (%) |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.7440 | 76.27 | 37.20 | 16.70 |
| Direct continuation · A/B | 0.6246 | 74.83 | 40.00 | 16.11 |
| SFT + traces · A/B | 0.5363 | 72.86 | 40.60 | 17.29 |
| Direct continuation · full/full | 0.6218 | 74.07 | 39.60 | 15.62 |
| SFT + traces · full/full | 0.5427 | 72.25 | 39.40 | 15.23 |

</div>

<figure class="l-body interleave-figure"><img src="/assets/blog/interleaving-pt-rl/llama-final-differences.svg" loading="lazy" alt="Final Llama accuracy differences from baseline on GSM8K, MATH-500, and Polaris. Every interleaved run improves MATH-500 and reduces GSM8K; only A/B trace training improves Polaris."><figcaption>Figure 8. Final Llama greedy accuracy relative to the baseline. No uncertainty intervals are shown.</figcaption></figure>

All four interleaved schedules finish higher than the baseline on MATH-500, by 2.2–3.4 points, and lower on GSM8K, by 1.4–4.0 points. Only the A/B trace run improves Polaris, the RL task, and the difference is only six questions out of 1,024 (177 vs. 171 correct). No schedule improves all three benchmarks. Each schedule is a single run, and we have no uncertainty estimate for these differences.

The interleaved runs also finish with a much lower Numina loss than the baseline, e.g., 0.5363 vs. 0.7440. This is not because SFT worked better: at the end of SFT, the same trace run actually has a slightly higher loss than the baseline (0.4534 vs. 0.4455). The gap comes from RL, which increases the loss on the SFT data. The baseline's loss rises from 0.4455 to 0.7440 over 3,000 consecutive RL updates, while the interleaved runs have had only 1,500 RL updates since their last SFT stage.

<h3>A reference point: RL, then traces, then RL</h3>

Some runs in the chess and math pretraining studies use a different schedule: they finish all pretraining first, then run RL, train only on the traces from that RL stage, and run RL again. Since there is no NTP data left when RL starts, these runs cannot tell us whether RL helps the model learn from NTP data. Instead, they test whether a round of training on the model's own correct rollouts helps the RL that follows, similar to Ring-Zero. We include them as a reference.

<div class="interleave-table" markdown="1">

| Study · schedule | Baseline final | Trace-only final | Difference |
| --- | ---: | ---: | ---: |
| Chess PT 256 × 8 · shuffled traces | 37.86 | 37.64 | −0.23 |
| Chess PT 256 × 8 · chronological traces | 37.86 | 38.49 | +0.62 |
| Chess PT 64 × 16 · A/B | 38.19 | 38.63 | +0.43 |
| Chess PT 64 × 16 · full/full | 38.19 | 38.20 | +0.00 |
| Math PT · A/B (SkyEasy) | 28.94 | 31.82 | +2.88 |

</div>

<p class="interleave-note">Final pass@1 (%). All chess intervals include zero. The math result is a single run.</p>

In chess, these runs finish level with the baseline. In math, the trace-only run finishes 2.88 points ahead on SkyEasy and 1.26 points ahead on MATH (32.60% vs. 31.34%), but 5.15 points behind on GSM8K (37.38% vs. 42.53%), with a higher pretraining loss (1.2513 vs. 1.2406). As with Llama, it gains on the RL task and similar problems but loses on another benchmark.

<details class="l-page" markdown="1"><summary>Full final-RL tables for the chess PT, math PT, and chess SFT studies</summary>

<div class="interleave-table" markdown="1">

| Chess PT, 256 × 8 · schedule | Pass@1 (%) | Pass@16 (%) |
| --- | ---: | ---: |
| Baseline | 37.86 | 54.80 |
| Two 2.5B schedules, no earlier RL | 37.94 | 55.88 |
| Traces, then plain PT2 (shuffled) | 36.61 | 55.74 |
| Traces, then plain PT2 (chronological) | 36.63 | 54.86 |
| PT2 + SFT + traces, stable mixing | 37.14 | 54.26 |
| PT2 + SFT + traces, global shuffle | 36.46 | 53.38 |
| Direct continuation | 36.36 | 54.26 |
| Direct continuation + late trace replay | 36.59 | 53.24 |
| Trace-only, shuffled | 37.64 | 53.51 |
| Trace-only, chronological | 38.49 | 53.85 |

</div>

<div class="interleave-table l-page" markdown="1">

| Chess PT, 64 × 16 · schedule | PT loss ↓ | Pass@1 (%) | Pass@2 (%) | Pass@4 (%) | Pass@8 (%) | Pass@16 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.534776 | 38.19 | 44.33 | 49.56 | 54.37 | 58.58 |
| Trace-only · A/B | 0.542875 | 38.63 | 44.73 | 49.75 | 54.30 | 58.58 |
| Trace-only · full/full | 0.543711 | 38.20 | 44.32 | 49.43 | 53.87 | 57.77 |
| PT + traces · A/B | 0.522745 | 36.16 | 42.52 | 47.91 | 52.64 | 56.96 |
| PT + traces · full/full | 0.520700 | 36.09 | 42.56 | 48.32 | 53.74 | 59.05 |

</div>

<div class="interleave-table l-page" markdown="1">

| OLMo2 1B · schedule | PT loss ↓ | SkyEasy pass@1 (%) | SkyEasy pass@16 (%) | GSM8K (%) | MATH (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1.240630 | 28.94 | 64.20 | 42.53 | 31.34 |
| Direct continuation, fresh AdamW | 1.242338 | 24.36 | 62.20 | 35.63 | 28.52 |
| Direct continuation, AdamW state kept | 1.241096 | 24.52 | 59.20 | 33.66 | 28.98 |
| PT + traces | 1.238009 | 24.89 | 61.60 | 31.31 | 28.98 |
| Trace-only · A/B | 1.251273 | 31.82 | 64.80 | 37.38 | 32.60 |

</div>

<div class="interleave-table l-page" markdown="1">

| Chess SFT · schedule | Held-out PT loss ↓ | Pass@1 (%) | Pass@2 (%) | Pass@4 (%) | Pass@8 (%) | Pass@16 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.7606 | 33.54 | 40.09 | 45.83 | 51.06 | 56.01 |
| Direct continuation · A/B | 0.8176 | 31.63 | 39.11 | 45.75 | 51.56 | 56.69 |
| SFT + traces · A/B | 0.7949 | 32.66 | 39.16 | 44.93 | 50.17 | 54.73 |
| Direct continuation · full/full | 0.7789 | 31.11 | 38.80 | 45.70 | 51.66 | 57.16 |
| SFT + traces · full/full | 0.7928 | 32.97 | 39.47 | 45.24 | 50.42 | 55.00 |

</div>

Six more 256 × 8 schedules vary how the optimizer state is carried between stages. Their final RL only runs on a larger set of 53,225 prompts, which the baseline does not use, so we list them in the appendix without comparing them to the baseline.

</details>

**Answer to the second question: not consistently.** Under the same RL budget, the baseline ends level with or ahead of the trace runs in four of five studies. Where interleaving does change the final model, it helps on some benchmarks and hurts on others: in Llama, it does better on MATH-500 and worse on GSM8K, and the math trace-only reference run shows the same pattern.

<h2 id="what-the-results-suggest">What the results suggest</h2>

We did not design experiments to explain these results, so the points below are our interpretation rather than tested conclusions.

**NTP has no incentive to keep what RL learned.** In NTP B, the model is only trained to predict the next token of a fixed dataset that was written before RL. Nothing in this objective rewards the model for keeping what RL learned, or penalizes it for moving away. If RL moves the model toward rollouts that are not in the NTP data, NTP pulls it back. This matches what we see: the RL gain only survives NTP when the RL traces are part of the NTP data.

**The gain is specific to the RL task.** Training on traces improves the RL task but slightly hurts the validation loss on the original data. In Llama and the math trace-only run, it also improves some benchmarks while hurting others. This suggests that RL, and training on its traces, mostly makes the model better at the specific problems being rewarded, rather than making it a better learner in general.

**The final RL seems to reach a similar level regardless of where it starts.** Starting the final RL at 33.5% pass@1 instead of 19.1% leaves RL less work to do, but does not raise the accuracy it eventually reaches. Our previous paper found that the accuracy after RL tracks the pretraining loss. None of the interleaved schedules lowers the pretraining loss, so a similar final accuracy is what we would expect. In math, where the trace run finishes clearly behind, its pretraining loss is also slightly higher than the baseline's.

**Splitting RL in half spends half of the budget on a model that gets changed afterwards.** The interleaved schedules spend half of their RL updates before NTP B. These updates are either forgotten during NTP B, or create a lead that the baseline's own RL catches up with anyway. The baseline spends all 3,000 updates on the model it ends up with.

Overall, alternating NTP and RL in the ways we tried does not make the two benefit each other. This is consistent with how other pipelines use this kind of alternation: as a practical tool to stabilize training, move progress to a newer base model, or control response length, rather than as a way to learn more from the same data.

<h2 id="related-work">Related work on mixing SFT and RL</h2>

A growing set of post-training methods combine SFT and RL in one run instead of running them in sequence, and several report gains over SFT followed by RL. Their setting differs from ours in two ways. First, the SFT data are demonstrations from a much stronger teacher, usually DeepSeek-R1 solutions, given to a 3B–8B model. Second, the method decides which questions or tokens receive supervision. Our NTP data come from the same distribution as the rest of training, and we apply no such selection. The ablations in these papers that remove the selection are therefore the closest published comparison to our schedules.

<div class="interleave-table interleave-text-table" markdown="1">

| Method | How SFT and RL are combined | Reported comparison |
| --- | --- | --- |
| [ReLIFT](https://arxiv.org/abs/2506.07527) | During GRPO, questions on which every rollout fails go into a buffer; when the buffer is full, one SFT step on R1 demonstrations for those questions | Qwen2.5-Math-7B, six-benchmark average: 52.6 vs 50.7 and 50.8 for SFT then RL (Table 1) |
| [SRFT](https://arxiv.org/abs/2506.19767) | One loss per step: entropy-weighted SFT on R1 demonstrations plus on-policy and off-policy RL terms | Qwen2.5-Math-7B, five math benchmarks: 59.5 vs 54.5 for SFT then RL (Table 2) |
| [BRIDGE](https://proceedings.mlr.press/v306/chen26an.html) | SFT and RL gradients combined at every step, with the SFT term reweighted by a meta-learned objective | Qwen3-8B-Base: 49.9 vs 45.5 for SFT then RL and 42.9 for RL only (Table 3) |
| [EvolveSearch](https://aclanthology.org/2025.emnlp-main.663.pdf) | Each round: SFT the base model on the correct rollouts collected so far, then GRPO on a new set of questions | Qwen2.5-7B search agent, after three rounds: +2.4 in-domain and +0.5 out-of-domain over RL only with the same questions (Table 1) |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | SFT → RL → SFT on 600k rejection-sampled reasoning traces plus 200k new non-reasoning examples → RL | AIME 2024: 79.8 vs 77.9 for RL only (R1-Zero); no controlled ablation of the alternation |

</div>

<p class="interleave-note">Numbers are as reported by each paper. None of the papers reports variance across seeds.</p>

When the selection rule is removed, these methods behave like our schedules. ReLIFT inserts the same number of SFT steps at a fixed interval of 8 RL steps instead of when hard questions accumulate. This lowers the average from 52.6 to 49.1, below SFT then RL. Filling the buffer with random questions instead of failed ones gives 45.5, and one SFT step after every RL step gives 23.8 ([ReLIFT, §4.3 and Table 4](https://arxiv.org/abs/2506.07527)). BRIDGE reports that adding the SFT and RL losses with fixed weights scores below RL alone: 29.4 vs 32.2 on Llama 3.2 3B Instruct and 40.1 vs 42.9 on Qwen3-8B-Base ([BRIDGE, Tables 2 and 3](https://proceedings.mlr.press/v306/chen26an.html)). In BRIDGE's runs, SRFT also scores below SFT then RL on both models, which differs from the gain SRFT reports for Qwen2.5-Math-7B. SRFT itself finds that running RL before SFT ends at 37.4, below SFT alone at 47.3 ([SRFT, Table 1](https://arxiv.org/abs/2506.19767)). A theoretical analysis reaches the same conclusion from the other side: RL raises the SFT loss and SFT lowers the RL reward, so the two stages cannot be optimized separately ([Niu et al., 2026](https://arxiv.org/abs/2601.07389)).

Two lines of work are closer to our pretraining studies. [RL Excursions during Pre-Training](https://arxiv.org/abs/2606.04272) applies RL to intermediate checkpoints of a 1B model pretrained from scratch and finds that RL is already effective early in pretraining. [RLP](https://arxiv.org/abs/2510.01265) uses an RL objective during pretraining itself, rewarding a sampled chain of thought by how much it raises the likelihood of the next token. Neither paper resumes ordinary pretraining after an RL stage and compares against the same data and RL budget without the interleaving. To our knowledge, the schedules in this post are the first such comparison.

Taken together, the published gains come from what the supervised data contain and from which examples receive supervision, not from alternation alone. Our studies hold both fixed: the NTP data are the same in every schedule, and every NTP example receives the same loss. Under those conditions we find no benefit, and the ablations above find none either.

<h2 id="open-questions">Open questions</h2>

We only tested a narrow set of schedules: one RL stage halfway through NTP, one return to NTP, and the same RL task before and after. Our conclusions apply to these schedules, not to every way of combining NTP and RL. Some directions we did not test:

- **A lower learning rate after RL.** We did not try continuing NTP B with a lower learning rate or a fast-decaying schedule. Smaller updates might erase less of what RL learned.
- **A larger RL learning rate in the middle.** The chess RL stages use a constant learning rate of 1e-5, and the math and Llama RL stages 1e-6. In a separate set of chess runs evaluated only on RL training prompts, raising the chess RL learning rate from 1e-5 to 1e-4 increased the relative change in the weights during RL from 0.2–0.4% to about 4%. We did not test the interleaved schedules at that learning rate on held-out data.
- **Supervision chosen by what RL fails on.** The methods in the previous section supervise the questions on which RL fails, using a stronger teacher's solutions. Our NTP B applies the same loss to every example. Selecting NTP B examples by the first RL stage's failures is the most direct next experiment.
- **Many short rounds.** Instead of one RL stage followed by one long NTP stage, the model could alternate many times, which is closer to how people learn. Each NTP stage would be shorter and might erase less of the RL stage before it.
- **Using RL to change what NTP learns.** Instead of mixing in RL traces, methods like ReMiT use an RL-tuned model to reweight tokens during NTP, which targets learning efficiency more directly.
- **Different tasks for RL and evaluation.** In our studies, the middle RL and the final RL train on the same task, so the middle RL mostly makes progress that the final RL would make anyway. Running the middle RL on a different task and measuring whether it helps later learning would be a cleaner test.
- **Measuring learning speed directly.** To show that NTP learns faster after RL, one would need to track the validation loss throughout NTP B rather than only at the end, compare against a run with the same data, optimizer, and learning-rate schedule, and count the FLOPs spent on RL and trace generation. We only measured the end points.
- **More seeds.** Most of our comparisons are single runs, and several final differences are only one or two points, small enough that a different seed could change them.

<h2 id="evaluation-details-and-limits">Evaluation details and limits</h2>

**Chess.** We evaluate on 1,480 held-out puzzles with 16 samples per puzzle at temperature 1 and top-p 1. The context is 2,048 tokens: at most 512 prompt tokens and 1,536 response tokens. Four of the original 1,484 puzzles exceed the prompt limit and are excluded. Pretraining loss uses 8,388,608 held-out target tokens. The chess SFT study uses a separate pretrained model with a corrected rotary positional embedding (RoPE) configuration, so its numbers should be compared only within that study. We record reasoning-format compliance separately from move correctness.

For $$N$$ problems with $$c_i$$ correct responses out of $$n=16$$ samples for problem $$i$$, we estimate

$$
\widehat{\operatorname{pass@}k}
=\frac{1}{N}\sum_{i=1}^{N}
\left[1-\frac{\binom{n-c_i}{k}}{\binom{n}{k}}\right],
\qquad 1\le k\le n,
$$

with $$\binom{n-c_i}{k}=0$$ when $$n-c_i<k$$. Pass@1 is mean sampled accuracy; pass@16 is the fraction of problems with at least one correct sample.

**OLMo2 math.** SkyEasy uses 500 questions with 16 samples each at temperature 1 and top-p 1, with the model's native 4,096-token context. GSM8K (1,319 questions) and MATH (the full 5,000-question test set, not MATH-500) use greedy decoding. Pretraining loss combines the pretraining evaluation components. The interleaved runs' NTP A is a 22.5B-token checkpoint; the first RL stage starts from it.

**Llama math.** We use the model's chat format and greedy decoding with an 8,192-token context. GSM8K has 1,319 questions, MATH-500 has 500, and held-out Polaris has 1,024. Polaris includes 42 questions with invalid reference answers, counted as failures for every model. Numina loss averages over 423,311 assistant tokens from 1,036 held-out records. SFT A and B contain 411,687 and 411,818 examples; RL A and B each contain 25,998 of the 51,996 Polaris prompts.

**Uncertainty.** Only the two chess pretraining studies have intervals, from 20,000 paired bootstrap resamples over puzzles. In the 256 × 8 study, the intervals were adjusted for 30 comparisons, including some runs we do not report here, so they are slightly wider than they need to be. In the 64 × 16 study they were adjusted for 8 comparisons. These intervals capture uncertainty from the choice of puzzles and samples, not from training with a different random seed. The math and SFT comparisons are single runs without intervals, so differences of one or two points there are not reliable.

**Budgets.** Within each study, all schedules use the same supervised data and the same total number of RL updates. Total compute differs, because trace runs generate and train on extra tokens and runs differ in optimizer resets and learning-rate schedules. A/B runs also change which prompts each RL stage sees. These are comparisons between whole training schedules and do not isolate each change within them.

**Learning rates.** RL uses a constant learning rate in every study: 1e-5 for the chess models, and 1e-6 for OLMo2 1B (after 50 warmup updates) and Llama 3.2 3B. NTP B starts a new learning-rate schedule with a new warmup. In chess pretraining the peak is 1e-3 with a fresh AdamW optimizer, in OLMo2 pretraining it is 4e-4, in chess SFT it is 1e-4, and in Llama SFT it is 2e-5 with a fresh AdamW optimizer. The math baseline runs its 45B pretraining tokens as one schedule, while the interleaved math runs use two 22.5B-token schedules. We have no math run that splits the schedule without RL, so in math we cannot separate the effect of RL in the middle from the effect of splitting the schedule. In the 256 × 8 chess study, splitting the schedule alone changed the final pass@1 by +0.08 points (37.94 vs 37.86).

**Which prompts we evaluate on.** Every chess number in this post is measured on 1,480 held-out puzzles that share no prompt with the RL data. Accuracy on the RL training prompts gives a different picture. For two of the 256 × 8 schedules, the difference in pass@1 was +9.08 points on the 28,419 RL training prompts but +0.63 points on the held-out puzzles, with an interval that includes zero. An earlier round of our experiments was evaluated only on the pool the RL prompts were drawn from. Its gains from interleaving were never measured on held-out puzzles, so we do not report those runs here.

<h2 id="complete-results">Complete results</h2>

These tables list every evaluated checkpoint for each study, as reported on the project dashboard. Values are rounded; the data files in the next section keep full precision.

<details><summary>Chess PT, 256 × 8: all 16 schedules before final RL, and the 10 with matched final RL</summary><p>Schedules 11–16 have final-RL results only on the larger 53,225-prompt set, which the baseline did not use. They are listed two tables below.</p><div class="interleave-table"><table>
<thead>
<tr>
<th>Checkpoint</th>
<th style="text-align: right;">PT loss ↓</th>
<th style="text-align: right;">Pass@1 (%)</th>
<th style="text-align: right;">Pass@2 (%)</th>
<th style="text-align: right;">Pass@4 (%)</th>
<th style="text-align: right;">Pass@8 (%)</th>
<th style="text-align: right;">Pass@16 (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Baseline: one 5B PT schedule · before final RL</td>
<td style="text-align: right;">0.512429</td>
<td style="text-align: right;">19.11</td>
<td style="text-align: right;">29.06</td>
<td style="text-align: right;">39.58</td>
<td style="text-align: right;">49.21</td>
<td style="text-align: right;">57.30</td>
</tr>
<tr>
<td>Baseline: one 5B PT schedule · after final RL</td>
<td style="text-align: right;">0.532375</td>
<td style="text-align: right;">37.86</td>
<td style="text-align: right;">43.04</td>
<td style="text-align: right;">47.35</td>
<td style="text-align: right;">51.18</td>
<td style="text-align: right;">54.80</td>
</tr>
<tr>
<td>Control: two 2.5B PT schedules · before final RL</td>
<td style="text-align: right;">0.507432</td>
<td style="text-align: right;">18.23</td>
<td style="text-align: right;">28.09</td>
<td style="text-align: right;">38.71</td>
<td style="text-align: right;">48.62</td>
<td style="text-align: right;">57.43</td>
</tr>
<tr>
<td>Control: two 2.5B PT schedules · after final RL</td>
<td style="text-align: right;">0.523177</td>
<td style="text-align: right;">37.94</td>
<td style="text-align: right;">43.18</td>
<td style="text-align: right;">47.76</td>
<td style="text-align: right;">52.03</td>
<td style="text-align: right;">55.88</td>
</tr>
<tr>
<td>Trace-only, shuffled · before final RL</td>
<td style="text-align: right;">0.537043</td>
<td style="text-align: right;">36.11</td>
<td style="text-align: right;">42.59</td>
<td style="text-align: right;">48.03</td>
<td style="text-align: right;">52.99</td>
<td style="text-align: right;">58.04</td>
</tr>
<tr>
<td>Trace-only, shuffled · after final RL</td>
<td style="text-align: right;">0.546173</td>
<td style="text-align: right;">37.64</td>
<td style="text-align: right;">42.69</td>
<td style="text-align: right;">46.90</td>
<td style="text-align: right;">50.42</td>
<td style="text-align: right;">53.51</td>
</tr>
<tr>
<td>Trace-only, chronological · before final RL</td>
<td style="text-align: right;">0.546131</td>
<td style="text-align: right;">37.70</td>
<td style="text-align: right;">43.32</td>
<td style="text-align: right;">47.97</td>
<td style="text-align: right;">52.03</td>
<td style="text-align: right;">55.88</td>
</tr>
<tr>
<td>Trace-only, chronological · after final RL</td>
<td style="text-align: right;">0.554020</td>
<td style="text-align: right;">38.49</td>
<td style="text-align: right;">43.36</td>
<td style="text-align: right;">47.43</td>
<td style="text-align: right;">50.91</td>
<td style="text-align: right;">53.85</td>
</tr>
<tr>
<td>Shuffled traces → PT2 · before final RL</td>
<td style="text-align: right;">0.507357</td>
<td style="text-align: right;">17.72</td>
<td style="text-align: right;">27.55</td>
<td style="text-align: right;">38.23</td>
<td style="text-align: right;">47.95</td>
<td style="text-align: right;">56.01</td>
</tr>
<tr>
<td>Shuffled traces → PT2 · after final RL</td>
<td style="text-align: right;">0.518310</td>
<td style="text-align: right;">36.61</td>
<td style="text-align: right;">42.34</td>
<td style="text-align: right;">47.23</td>
<td style="text-align: right;">51.63</td>
<td style="text-align: right;">55.74</td>
</tr>
<tr>
<td>Chronological traces → PT2 · before final RL</td>
<td style="text-align: right;">0.507585</td>
<td style="text-align: right;">17.82</td>
<td style="text-align: right;">27.58</td>
<td style="text-align: right;">38.26</td>
<td style="text-align: right;">48.25</td>
<td style="text-align: right;">56.89</td>
</tr>
<tr>
<td>Chronological traces → PT2 · after final RL</td>
<td style="text-align: right;">0.517506</td>
<td style="text-align: right;">36.63</td>
<td style="text-align: right;">42.22</td>
<td style="text-align: right;">46.78</td>
<td style="text-align: right;">50.83</td>
<td style="text-align: right;">54.86</td>
</tr>
<tr>
<td>PT2 + traces, stable mixing · before final RL</td>
<td style="text-align: right;">0.510979</td>
<td style="text-align: right;">33.50</td>
<td style="text-align: right;">40.03</td>
<td style="text-align: right;">45.52</td>
<td style="text-align: right;">50.45</td>
<td style="text-align: right;">55.00</td>
</tr>
<tr>
<td>PT2 + traces, stable mixing · after final RL</td>
<td style="text-align: right;">0.520220</td>
<td style="text-align: right;">37.14</td>
<td style="text-align: right;">42.12</td>
<td style="text-align: right;">46.51</td>
<td style="text-align: right;">50.56</td>
<td style="text-align: right;">54.26</td>
</tr>
<tr>
<td>PT2 + traces, global shuffle · before final RL</td>
<td style="text-align: right;">0.509731</td>
<td style="text-align: right;">32.57</td>
<td style="text-align: right;">39.48</td>
<td style="text-align: right;">45.39</td>
<td style="text-align: right;">50.69</td>
<td style="text-align: right;">55.27</td>
</tr>
<tr>
<td>PT2 + traces, global shuffle · after final RL</td>
<td style="text-align: right;">0.519042</td>
<td style="text-align: right;">36.46</td>
<td style="text-align: right;">41.21</td>
<td style="text-align: right;">45.40</td>
<td style="text-align: right;">49.32</td>
<td style="text-align: right;">53.38</td>
</tr>
<tr>
<td>RL weights → PT2, fresh AdamW · before final RL</td>
<td style="text-align: right;">0.507826</td>
<td style="text-align: right;">17.85</td>
<td style="text-align: right;">27.68</td>
<td style="text-align: right;">38.27</td>
<td style="text-align: right;">47.86</td>
<td style="text-align: right;">55.68</td>
</tr>
<tr>
<td>RL weights → PT2, fresh AdamW · after final RL</td>
<td style="text-align: right;">0.519173</td>
<td style="text-align: right;">36.36</td>
<td style="text-align: right;">42.15</td>
<td style="text-align: right;">46.98</td>
<td style="text-align: right;">50.94</td>
<td style="text-align: right;">54.26</td>
</tr>
<tr>
<td>RL weights → PT2 + late trace replay · before final RL</td>
<td style="text-align: right;">0.508571</td>
<td style="text-align: right;">26.34</td>
<td style="text-align: right;">35.72</td>
<td style="text-align: right;">43.99</td>
<td style="text-align: right;">51.20</td>
<td style="text-align: right;">57.84</td>
</tr>
<tr>
<td>RL weights → PT2 + late trace replay · after final RL</td>
<td style="text-align: right;">0.519834</td>
<td style="text-align: right;">36.59</td>
<td style="text-align: right;">41.77</td>
<td style="text-align: right;">45.96</td>
<td style="text-align: right;">49.66</td>
<td style="text-align: right;">53.24</td>
</tr>
<tr>
<td>RL weights → PT2, carry AdamW · before final RL</td>
<td style="text-align: right;">0.507727</td>
<td style="text-align: right;">18.11</td>
<td style="text-align: right;">27.98</td>
<td style="text-align: right;">38.53</td>
<td style="text-align: right;">48.07</td>
<td style="text-align: right;">56.49</td>
</tr>
<tr>
<td>RL weights → PT2 + late replay, carry AdamW · before final RL</td>
<td style="text-align: right;">0.507952</td>
<td style="text-align: right;">26.22</td>
<td style="text-align: right;">35.80</td>
<td style="text-align: right;">44.13</td>
<td style="text-align: right;">51.45</td>
<td style="text-align: right;">58.18</td>
</tr>
<tr>
<td>PT weights + PT AdamW → PT2 + traces, ordered · before final RL</td>
<td style="text-align: right;">0.511257</td>
<td style="text-align: right;">33.56</td>
<td style="text-align: right;">40.14</td>
<td style="text-align: right;">45.71</td>
<td style="text-align: right;">50.49</td>
<td style="text-align: right;">54.66</td>
</tr>
<tr>
<td>PT weights + PT AdamW → PT2 + traces, shuffled · before final RL</td>
<td style="text-align: right;">0.510219</td>
<td style="text-align: right;">32.81</td>
<td style="text-align: right;">39.59</td>
<td style="text-align: right;">45.35</td>
<td style="text-align: right;">50.63</td>
<td style="text-align: right;">55.81</td>
</tr>
<tr>
<td>RL weights + RL AdamW → PT2 + all traces · before final RL</td>
<td style="text-align: right;">0.511704</td>
<td style="text-align: right;">34.08</td>
<td style="text-align: right;">40.40</td>
<td style="text-align: right;">45.72</td>
<td style="text-align: right;">50.67</td>
<td style="text-align: right;">55.61</td>
</tr>
<tr>
<td>RL weights + fresh AdamW → PT2 + all traces · before final RL</td>
<td style="text-align: right;">0.510470</td>
<td style="text-align: right;">34.04</td>
<td style="text-align: right;">40.55</td>
<td style="text-align: right;">45.99</td>
<td style="text-align: right;">50.95</td>
<td style="text-align: right;">55.41</td>
</tr>
</tbody>
</table></div></details>
<details><summary>Chess PT, 64 × 16: all five schedules, before and after final RL</summary><div class="interleave-table"><table>
<thead>
<tr>
<th>Checkpoint</th>
<th style="text-align: right;">PT loss ↓</th>
<th style="text-align: right;">Pass@1 (%)</th>
<th style="text-align: right;">Pass@2 (%)</th>
<th style="text-align: right;">Pass@4 (%)</th>
<th style="text-align: right;">Pass@8 (%)</th>
<th style="text-align: right;">Pass@16 (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Baseline / full · before final RL</td>
<td style="text-align: right;">0.512429</td>
<td style="text-align: right;">19.11</td>
<td style="text-align: right;">29.06</td>
<td style="text-align: right;">39.58</td>
<td style="text-align: right;">49.21</td>
<td style="text-align: right;">57.30</td>
</tr>
<tr>
<td>Baseline / full · after final RL</td>
<td style="text-align: right;">0.534776</td>
<td style="text-align: right;">38.19</td>
<td style="text-align: right;">44.33</td>
<td style="text-align: right;">49.56</td>
<td style="text-align: right;">54.37</td>
<td style="text-align: right;">58.58</td>
</tr>
<tr>
<td>Trace-only / A→B · before final RL</td>
<td style="text-align: right;">0.538013</td>
<td style="text-align: right;">36.79</td>
<td style="text-align: right;">43.46</td>
<td style="text-align: right;">49.12</td>
<td style="text-align: right;">54.20</td>
<td style="text-align: right;">58.78</td>
</tr>
<tr>
<td>Trace-only / A→B · after final RL</td>
<td style="text-align: right;">0.542875</td>
<td style="text-align: right;">38.63</td>
<td style="text-align: right;">44.73</td>
<td style="text-align: right;">49.75</td>
<td style="text-align: right;">54.30</td>
<td style="text-align: right;">58.58</td>
</tr>
<tr>
<td>Trace-only / full→full · before final RL</td>
<td style="text-align: right;">0.535673</td>
<td style="text-align: right;">36.59</td>
<td style="text-align: right;">43.79</td>
<td style="text-align: right;">49.64</td>
<td style="text-align: right;">54.61</td>
<td style="text-align: right;">59.05</td>
</tr>
<tr>
<td>Trace-only / full→full · after final RL</td>
<td style="text-align: right;">0.543711</td>
<td style="text-align: right;">38.20</td>
<td style="text-align: right;">44.32</td>
<td style="text-align: right;">49.43</td>
<td style="text-align: right;">53.87</td>
<td style="text-align: right;">57.77</td>
</tr>
<tr>
<td>PT + traces / A→B · before final RL</td>
<td style="text-align: right;">0.509955</td>
<td style="text-align: right;">29.47</td>
<td style="text-align: right;">38.18</td>
<td style="text-align: right;">45.90</td>
<td style="text-align: right;">52.80</td>
<td style="text-align: right;">59.05</td>
</tr>
<tr>
<td>PT + traces / A→B · after final RL</td>
<td style="text-align: right;">0.522745</td>
<td style="text-align: right;">36.16</td>
<td style="text-align: right;">42.52</td>
<td style="text-align: right;">47.91</td>
<td style="text-align: right;">52.64</td>
<td style="text-align: right;">56.96</td>
</tr>
<tr>
<td>PT + traces / full→full · before final RL</td>
<td style="text-align: right;">0.509888</td>
<td style="text-align: right;">30.24</td>
<td style="text-align: right;">39.10</td>
<td style="text-align: right;">46.80</td>
<td style="text-align: right;">53.40</td>
<td style="text-align: right;">59.12</td>
</tr>
<tr>
<td>PT + traces / full→full · after final RL</td>
<td style="text-align: right;">0.520700</td>
<td style="text-align: right;">36.09</td>
<td style="text-align: right;">42.56</td>
<td style="text-align: right;">48.32</td>
<td style="text-align: right;">53.74</td>
<td style="text-align: right;">59.05</td>
</tr>
</tbody>
</table></div></details>
<details><summary>Chess PT, 256 × 8: final RL on the larger 53,225-prompt set, schedules 11–16</summary><p>These runs use 53,225 RL training prompts. The main comparison uses 28,419 prompts, so we report these results separately.</p><div class="interleave-table"><table>
<thead>
<tr>
<th>Checkpoint</th>
<th style="text-align: right;">PT loss ↓</th>
<th style="text-align: right;">Pass@1 (%)</th>
<th style="text-align: right;">Pass@2 (%)</th>
<th style="text-align: right;">Pass@4 (%)</th>
<th style="text-align: right;">Pass@8 (%)</th>
<th style="text-align: right;">Pass@16 (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>rl11_final · 53,225 RL prompts</td>
<td style="text-align: right;">0.520187</td>
<td style="text-align: right;">36.80</td>
<td style="text-align: right;">43.68</td>
<td style="text-align: right;">49.47</td>
<td style="text-align: right;">54.64</td>
<td style="text-align: right;">59.59</td>
</tr>
<tr>
<td>rl12_final · 53,225 RL prompts</td>
<td style="text-align: right;">0.518223</td>
<td style="text-align: right;">37.39</td>
<td style="text-align: right;">43.10</td>
<td style="text-align: right;">48.09</td>
<td style="text-align: right;">52.65</td>
<td style="text-align: right;">56.76</td>
</tr>
<tr>
<td>rl13_final · 53,225 RL prompts</td>
<td style="text-align: right;">0.519089</td>
<td style="text-align: right;">37.82</td>
<td style="text-align: right;">43.15</td>
<td style="text-align: right;">47.64</td>
<td style="text-align: right;">51.62</td>
<td style="text-align: right;">55.61</td>
</tr>
<tr>
<td>rl14_final · 53,225 RL prompts</td>
<td style="text-align: right;">0.519770</td>
<td style="text-align: right;">37.96</td>
<td style="text-align: right;">43.29</td>
<td style="text-align: right;">47.95</td>
<td style="text-align: right;">52.17</td>
<td style="text-align: right;">55.95</td>
</tr>
<tr>
<td>rl15_final · 53,225 RL prompts</td>
<td style="text-align: right;">0.521709</td>
<td style="text-align: right;">38.81</td>
<td style="text-align: right;">44.09</td>
<td style="text-align: right;">48.47</td>
<td style="text-align: right;">52.29</td>
<td style="text-align: right;">55.68</td>
</tr>
<tr>
<td>rl16_final · 53,225 RL prompts</td>
<td style="text-align: right;">0.518890</td>
<td style="text-align: right;">38.04</td>
<td style="text-align: right;">43.33</td>
<td style="text-align: right;">47.82</td>
<td style="text-align: right;">51.93</td>
<td style="text-align: right;">55.95</td>
</tr>
</tbody>
</table></div></details>
<details><summary>Chess pretraining, 256 × 8: all evaluated checkpoints · full pass@1–16</summary><div class="interleave-table"><table>
<thead>
<tr>
<th>Checkpoint</th>
<th style="text-align: right;">PT loss ↓</th>
<th style="text-align: right;">Pass@1 (%)</th>
<th style="text-align: right;">Pass@2 (%)</th>
<th style="text-align: right;">Pass@3 (%)</th>
<th style="text-align: right;">Pass@4 (%)</th>
<th style="text-align: right;">Pass@5 (%)</th>
<th style="text-align: right;">Pass@6 (%)</th>
<th style="text-align: right;">Pass@7 (%)</th>
<th style="text-align: right;">Pass@8 (%)</th>
<th style="text-align: right;">Pass@9 (%)</th>
<th style="text-align: right;">Pass@10 (%)</th>
<th style="text-align: right;">Pass@11 (%)</th>
<th style="text-align: right;">Pass@12 (%)</th>
<th style="text-align: right;">Pass@13 (%)</th>
<th style="text-align: right;">Pass@14 (%)</th>
<th style="text-align: right;">Pass@15 (%)</th>
<th style="text-align: right;">Pass@16 (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>pt5b_sft3_full</td>
<td style="text-align: right;">0.512429</td>
<td style="text-align: right;">19.11</td>
<td style="text-align: right;">29.06</td>
<td style="text-align: right;">35.26</td>
<td style="text-align: right;">39.58</td>
<td style="text-align: right;">42.83</td>
<td style="text-align: right;">45.39</td>
<td style="text-align: right;">47.47</td>
<td style="text-align: right;">49.21</td>
<td style="text-align: right;">50.70</td>
<td style="text-align: right;">51.98</td>
<td style="text-align: right;">53.12</td>
<td style="text-align: right;">54.12</td>
<td style="text-align: right;">55.03</td>
<td style="text-align: right;">55.85</td>
<td style="text-align: right;">56.60</td>
<td style="text-align: right;">57.30</td>
</tr>
<tr>
<td>pt2p5b_sft3_first</td>
<td style="text-align: right;">0.525055</td>
<td style="text-align: right;">13.85</td>
<td style="text-align: right;">22.14</td>
<td style="text-align: right;">27.84</td>
<td style="text-align: right;">32.05</td>
<td style="text-align: right;">35.33</td>
<td style="text-align: right;">37.99</td>
<td style="text-align: right;">40.19</td>
<td style="text-align: right;">42.07</td>
<td style="text-align: right;">43.69</td>
<td style="text-align: right;">45.11</td>
<td style="text-align: right;">46.37</td>
<td style="text-align: right;">47.49</td>
<td style="text-align: right;">48.49</td>
<td style="text-align: right;">49.41</td>
<td style="text-align: right;">50.24</td>
<td style="text-align: right;">51.01</td>
</tr>
<tr>
<td>pt2p5b_sft3_second_plain</td>
<td style="text-align: right;">0.507432</td>
<td style="text-align: right;">18.23</td>
<td style="text-align: right;">28.09</td>
<td style="text-align: right;">34.33</td>
<td style="text-align: right;">38.71</td>
<td style="text-align: right;">42.01</td>
<td style="text-align: right;">44.64</td>
<td style="text-align: right;">46.80</td>
<td style="text-align: right;">48.62</td>
<td style="text-align: right;">50.19</td>
<td style="text-align: right;">51.58</td>
<td style="text-align: right;">52.80</td>
<td style="text-align: right;">53.91</td>
<td style="text-align: right;">54.91</td>
<td style="text-align: right;">55.82</td>
<td style="text-align: right;">56.66</td>
<td style="text-align: right;">57.43</td>
</tr>
<tr>
<td>rl_pt5b_source</td>
<td style="text-align: right;">0.527518</td>
<td style="text-align: right;">36.93</td>
<td style="text-align: right;">42.50</td>
<td style="text-align: right;">45.34</td>
<td style="text-align: right;">47.24</td>
<td style="text-align: right;">48.64</td>
<td style="text-align: right;">49.75</td>
<td style="text-align: right;">50.65</td>
<td style="text-align: right;">51.42</td>
<td style="text-align: right;">52.08</td>
<td style="text-align: right;">52.66</td>
<td style="text-align: right;">53.18</td>
<td style="text-align: right;">53.64</td>
<td style="text-align: right;">54.06</td>
<td style="text-align: right;">54.45</td>
<td style="text-align: right;">54.81</td>
<td style="text-align: right;">55.14</td>
</tr>
<tr>
<td>rl_pt2p5b_source</td>
<td style="text-align: right;">0.540289</td>
<td style="text-align: right;">30.79</td>
<td style="text-align: right;">36.39</td>
<td style="text-align: right;">39.32</td>
<td style="text-align: right;">41.26</td>
<td style="text-align: right;">42.71</td>
<td style="text-align: right;">43.85</td>
<td style="text-align: right;">44.80</td>
<td style="text-align: right;">45.60</td>
<td style="text-align: right;">46.30</td>
<td style="text-align: right;">46.91</td>
<td style="text-align: right;">47.46</td>
<td style="text-align: right;">47.95</td>
<td style="text-align: right;">48.40</td>
<td style="text-align: right;">48.81</td>
<td style="text-align: right;">49.18</td>
<td style="text-align: right;">49.53</td>
</tr>
<tr>
<td>trace03_shuffled</td>
<td style="text-align: right;">0.537043</td>
<td style="text-align: right;">36.11</td>
<td style="text-align: right;">42.59</td>
<td style="text-align: right;">45.86</td>
<td style="text-align: right;">48.03</td>
<td style="text-align: right;">49.66</td>
<td style="text-align: right;">50.96</td>
<td style="text-align: right;">52.05</td>
<td style="text-align: right;">52.99</td>
<td style="text-align: right;">53.83</td>
<td style="text-align: right;">54.58</td>
<td style="text-align: right;">55.27</td>
<td style="text-align: right;">55.90</td>
<td style="text-align: right;">56.49</td>
<td style="text-align: right;">57.04</td>
<td style="text-align: right;">57.55</td>
<td style="text-align: right;">58.04</td>
</tr>
<tr>
<td>trace04_chronological</td>
<td style="text-align: right;">0.546131</td>
<td style="text-align: right;">37.70</td>
<td style="text-align: right;">43.32</td>
<td style="text-align: right;">46.13</td>
<td style="text-align: right;">47.97</td>
<td style="text-align: right;">49.33</td>
<td style="text-align: right;">50.40</td>
<td style="text-align: right;">51.28</td>
<td style="text-align: right;">52.03</td>
<td style="text-align: right;">52.69</td>
<td style="text-align: right;">53.27</td>
<td style="text-align: right;">53.80</td>
<td style="text-align: right;">54.28</td>
<td style="text-align: right;">54.72</td>
<td style="text-align: right;">55.13</td>
<td style="text-align: right;">55.52</td>
<td style="text-align: right;">55.88</td>
</tr>
<tr>
<td>trace05_shuffled</td>
<td style="text-align: right;">0.552901</td>
<td style="text-align: right;">30.27</td>
<td style="text-align: right;">36.64</td>
<td style="text-align: right;">40.01</td>
<td style="text-align: right;">42.27</td>
<td style="text-align: right;">43.94</td>
<td style="text-align: right;">45.24</td>
<td style="text-align: right;">46.31</td>
<td style="text-align: right;">47.20</td>
<td style="text-align: right;">47.97</td>
<td style="text-align: right;">48.63</td>
<td style="text-align: right;">49.23</td>
<td style="text-align: right;">49.76</td>
<td style="text-align: right;">50.24</td>
<td style="text-align: right;">50.68</td>
<td style="text-align: right;">51.10</td>
<td style="text-align: right;">51.49</td>
</tr>
<tr>
<td>trace06_chronological</td>
<td style="text-align: right;">0.560367</td>
<td style="text-align: right;">30.74</td>
<td style="text-align: right;">36.10</td>
<td style="text-align: right;">38.93</td>
<td style="text-align: right;">40.84</td>
<td style="text-align: right;">42.30</td>
<td style="text-align: right;">43.47</td>
<td style="text-align: right;">44.45</td>
<td style="text-align: right;">45.30</td>
<td style="text-align: right;">46.05</td>
<td style="text-align: right;">46.72</td>
<td style="text-align: right;">47.32</td>
<td style="text-align: right;">47.87</td>
<td style="text-align: right;">48.37</td>
<td style="text-align: right;">48.84</td>
<td style="text-align: right;">49.27</td>
<td style="text-align: right;">49.66</td>
</tr>
<tr>
<td>trace05_then_pt_second</td>
<td style="text-align: right;">0.507357</td>
<td style="text-align: right;">17.72</td>
<td style="text-align: right;">27.55</td>
<td style="text-align: right;">33.83</td>
<td style="text-align: right;">38.23</td>
<td style="text-align: right;">41.52</td>
<td style="text-align: right;">44.10</td>
<td style="text-align: right;">46.20</td>
<td style="text-align: right;">47.95</td>
<td style="text-align: right;">49.45</td>
<td style="text-align: right;">50.74</td>
<td style="text-align: right;">51.87</td>
<td style="text-align: right;">52.88</td>
<td style="text-align: right;">53.78</td>
<td style="text-align: right;">54.60</td>
<td style="text-align: right;">55.34</td>
<td style="text-align: right;">56.01</td>
</tr>
<tr>
<td>trace06_then_pt_second</td>
<td style="text-align: right;">0.507585</td>
<td style="text-align: right;">17.82</td>
<td style="text-align: right;">27.58</td>
<td style="text-align: right;">33.84</td>
<td style="text-align: right;">38.26</td>
<td style="text-align: right;">41.61</td>
<td style="text-align: right;">44.26</td>
<td style="text-align: right;">46.43</td>
<td style="text-align: right;">48.25</td>
<td style="text-align: right;">49.82</td>
<td style="text-align: right;">51.18</td>
<td style="text-align: right;">52.39</td>
<td style="text-align: right;">53.47</td>
<td style="text-align: right;">54.45</td>
<td style="text-align: right;">55.33</td>
<td style="text-align: right;">56.14</td>
<td style="text-align: right;">56.89</td>
</tr>
<tr>
<td>trace07_mixed_pt_second</td>
<td style="text-align: right;">0.510979</td>
<td style="text-align: right;">33.50</td>
<td style="text-align: right;">40.03</td>
<td style="text-align: right;">43.33</td>
<td style="text-align: right;">45.52</td>
<td style="text-align: right;">47.16</td>
<td style="text-align: right;">48.45</td>
<td style="text-align: right;">49.53</td>
<td style="text-align: right;">50.45</td>
<td style="text-align: right;">51.24</td>
<td style="text-align: right;">51.95</td>
<td style="text-align: right;">52.58</td>
<td style="text-align: right;">53.16</td>
<td style="text-align: right;">53.68</td>
<td style="text-align: right;">54.15</td>
<td style="text-align: right;">54.59</td>
<td style="text-align: right;">55.00</td>
</tr>
<tr>
<td>trace08_global_shuffle_pt_second</td>
<td style="text-align: right;">0.509731</td>
<td style="text-align: right;">32.57</td>
<td style="text-align: right;">39.48</td>
<td style="text-align: right;">43.02</td>
<td style="text-align: right;">45.39</td>
<td style="text-align: right;">47.16</td>
<td style="text-align: right;">48.56</td>
<td style="text-align: right;">49.72</td>
<td style="text-align: right;">50.69</td>
<td style="text-align: right;">51.52</td>
<td style="text-align: right;">52.25</td>
<td style="text-align: right;">52.89</td>
<td style="text-align: right;">53.47</td>
<td style="text-align: right;">53.98</td>
<td style="text-align: right;">54.45</td>
<td style="text-align: right;">54.88</td>
<td style="text-align: right;">55.27</td>
</tr>
<tr>
<td>rl01_final</td>
<td style="text-align: right;">0.532375</td>
<td style="text-align: right;">37.86</td>
<td style="text-align: right;">43.04</td>
<td style="text-align: right;">45.63</td>
<td style="text-align: right;">47.35</td>
<td style="text-align: right;">48.63</td>
<td style="text-align: right;">49.64</td>
<td style="text-align: right;">50.47</td>
<td style="text-align: right;">51.18</td>
<td style="text-align: right;">51.80</td>
<td style="text-align: right;">52.35</td>
<td style="text-align: right;">52.84</td>
<td style="text-align: right;">53.29</td>
<td style="text-align: right;">53.71</td>
<td style="text-align: right;">54.09</td>
<td style="text-align: right;">54.46</td>
<td style="text-align: right;">54.80</td>
</tr>
<tr>
<td>rl02_final</td>
<td style="text-align: right;">0.523177</td>
<td style="text-align: right;">37.94</td>
<td style="text-align: right;">43.18</td>
<td style="text-align: right;">45.90</td>
<td style="text-align: right;">47.76</td>
<td style="text-align: right;">49.16</td>
<td style="text-align: right;">50.29</td>
<td style="text-align: right;">51.23</td>
<td style="text-align: right;">52.03</td>
<td style="text-align: right;">52.72</td>
<td style="text-align: right;">53.32</td>
<td style="text-align: right;">53.86</td>
<td style="text-align: right;">54.35</td>
<td style="text-align: right;">54.78</td>
<td style="text-align: right;">55.18</td>
<td style="text-align: right;">55.54</td>
<td style="text-align: right;">55.88</td>
</tr>
<tr>
<td>rl03_final</td>
<td style="text-align: right;">0.546173</td>
<td style="text-align: right;">37.64</td>
<td style="text-align: right;">42.69</td>
<td style="text-align: right;">45.24</td>
<td style="text-align: right;">46.90</td>
<td style="text-align: right;">48.10</td>
<td style="text-align: right;">49.03</td>
<td style="text-align: right;">49.79</td>
<td style="text-align: right;">50.42</td>
<td style="text-align: right;">50.97</td>
<td style="text-align: right;">51.46</td>
<td style="text-align: right;">51.89</td>
<td style="text-align: right;">52.27</td>
<td style="text-align: right;">52.62</td>
<td style="text-align: right;">52.94</td>
<td style="text-align: right;">53.24</td>
<td style="text-align: right;">53.51</td>
</tr>
<tr>
<td>rl04_final</td>
<td style="text-align: right;">0.554020</td>
<td style="text-align: right;">38.49</td>
<td style="text-align: right;">43.36</td>
<td style="text-align: right;">45.83</td>
<td style="text-align: right;">47.43</td>
<td style="text-align: right;">48.61</td>
<td style="text-align: right;">49.53</td>
<td style="text-align: right;">50.28</td>
<td style="text-align: right;">50.91</td>
<td style="text-align: right;">51.45</td>
<td style="text-align: right;">51.92</td>
<td style="text-align: right;">52.33</td>
<td style="text-align: right;">52.70</td>
<td style="text-align: right;">53.03</td>
<td style="text-align: right;">53.32</td>
<td style="text-align: right;">53.60</td>
<td style="text-align: right;">53.85</td>
</tr>
<tr>
<td>rl05_final</td>
<td style="text-align: right;">0.518310</td>
<td style="text-align: right;">36.61</td>
<td style="text-align: right;">42.34</td>
<td style="text-align: right;">45.27</td>
<td style="text-align: right;">47.23</td>
<td style="text-align: right;">48.69</td>
<td style="text-align: right;">49.85</td>
<td style="text-align: right;">50.81</td>
<td style="text-align: right;">51.63</td>
<td style="text-align: right;">52.34</td>
<td style="text-align: right;">52.97</td>
<td style="text-align: right;">53.53</td>
<td style="text-align: right;">54.05</td>
<td style="text-align: right;">54.52</td>
<td style="text-align: right;">54.96</td>
<td style="text-align: right;">55.36</td>
<td style="text-align: right;">55.74</td>
</tr>
<tr>
<td>rl06_final</td>
<td style="text-align: right;">0.517506</td>
<td style="text-align: right;">36.63</td>
<td style="text-align: right;">42.22</td>
<td style="text-align: right;">44.98</td>
<td style="text-align: right;">46.78</td>
<td style="text-align: right;">48.11</td>
<td style="text-align: right;">49.17</td>
<td style="text-align: right;">50.06</td>
<td style="text-align: right;">50.83</td>
<td style="text-align: right;">51.52</td>
<td style="text-align: right;">52.13</td>
<td style="text-align: right;">52.69</td>
<td style="text-align: right;">53.20</td>
<td style="text-align: right;">53.67</td>
<td style="text-align: right;">54.10</td>
<td style="text-align: right;">54.50</td>
<td style="text-align: right;">54.86</td>
</tr>
<tr>
<td>rl07_final</td>
<td style="text-align: right;">0.520220</td>
<td style="text-align: right;">37.14</td>
<td style="text-align: right;">42.12</td>
<td style="text-align: right;">44.74</td>
<td style="text-align: right;">46.51</td>
<td style="text-align: right;">47.85</td>
<td style="text-align: right;">48.92</td>
<td style="text-align: right;">49.81</td>
<td style="text-align: right;">50.56</td>
<td style="text-align: right;">51.22</td>
<td style="text-align: right;">51.80</td>
<td style="text-align: right;">52.31</td>
<td style="text-align: right;">52.77</td>
<td style="text-align: right;">53.19</td>
<td style="text-align: right;">53.58</td>
<td style="text-align: right;">53.93</td>
<td style="text-align: right;">54.26</td>
</tr>
<tr>
<td>rl08_final</td>
<td style="text-align: right;">0.519042</td>
<td style="text-align: right;">36.46</td>
<td style="text-align: right;">41.21</td>
<td style="text-align: right;">43.72</td>
<td style="text-align: right;">45.40</td>
<td style="text-align: right;">46.68</td>
<td style="text-align: right;">47.70</td>
<td style="text-align: right;">48.57</td>
<td style="text-align: right;">49.32</td>
<td style="text-align: right;">49.99</td>
<td style="text-align: right;">50.59</td>
<td style="text-align: right;">51.14</td>
<td style="text-align: right;">51.65</td>
<td style="text-align: right;">52.13</td>
<td style="text-align: right;">52.57</td>
<td style="text-align: right;">52.99</td>
<td style="text-align: right;">53.38</td>
</tr>
<tr>
<td>trace09_direct_rl_then_pt_second_v3</td>
<td style="text-align: right;">0.507826</td>
<td style="text-align: right;">17.85</td>
<td style="text-align: right;">27.68</td>
<td style="text-align: right;">33.91</td>
<td style="text-align: right;">38.27</td>
<td style="text-align: right;">41.53</td>
<td style="text-align: right;">44.08</td>
<td style="text-align: right;">46.14</td>
<td style="text-align: right;">47.86</td>
<td style="text-align: right;">49.32</td>
<td style="text-align: right;">50.58</td>
<td style="text-align: right;">51.68</td>
<td style="text-align: right;">52.65</td>
<td style="text-align: right;">53.53</td>
<td style="text-align: right;">54.31</td>
<td style="text-align: right;">55.03</td>
<td style="text-align: right;">55.68</td>
</tr>
<tr>
<td>trace10_direct_rl_then_pt_replay_second_v2</td>
<td style="text-align: right;">0.508571</td>
<td style="text-align: right;">26.34</td>
<td style="text-align: right;">35.72</td>
<td style="text-align: right;">40.71</td>
<td style="text-align: right;">43.99</td>
<td style="text-align: right;">46.40</td>
<td style="text-align: right;">48.30</td>
<td style="text-align: right;">49.87</td>
<td style="text-align: right;">51.20</td>
<td style="text-align: right;">52.36</td>
<td style="text-align: right;">53.40</td>
<td style="text-align: right;">54.32</td>
<td style="text-align: right;">55.16</td>
<td style="text-align: right;">55.92</td>
<td style="text-align: right;">56.61</td>
<td style="text-align: right;">57.25</td>
<td style="text-align: right;">57.84</td>
</tr>
<tr>
<td>rl11_12_shared_continuous</td>
<td style="text-align: right;">0.539732</td>
<td style="text-align: right;">31.53</td>
<td style="text-align: right;">36.98</td>
<td style="text-align: right;">39.85</td>
<td style="text-align: right;">41.79</td>
<td style="text-align: right;">43.24</td>
<td style="text-align: right;">44.39</td>
<td style="text-align: right;">45.33</td>
<td style="text-align: right;">46.12</td>
<td style="text-align: right;">46.80</td>
<td style="text-align: right;">47.41</td>
<td style="text-align: right;">47.94</td>
<td style="text-align: right;">48.43</td>
<td style="text-align: right;">48.88</td>
<td style="text-align: right;">49.30</td>
<td style="text-align: right;">49.70</td>
<td style="text-align: right;">50.07</td>
</tr>
<tr>
<td>trace11_continuous_then_pt_second</td>
<td style="text-align: right;">0.507727</td>
<td style="text-align: right;">18.11</td>
<td style="text-align: right;">27.98</td>
<td style="text-align: right;">34.20</td>
<td style="text-align: right;">38.53</td>
<td style="text-align: right;">41.75</td>
<td style="text-align: right;">44.28</td>
<td style="text-align: right;">46.34</td>
<td style="text-align: right;">48.07</td>
<td style="text-align: right;">49.56</td>
<td style="text-align: right;">50.87</td>
<td style="text-align: right;">52.03</td>
<td style="text-align: right;">53.08</td>
<td style="text-align: right;">54.04</td>
<td style="text-align: right;">54.92</td>
<td style="text-align: right;">55.73</td>
<td style="text-align: right;">56.49</td>
</tr>
<tr>
<td>trace12_continuous_then_pt_replay_second</td>
<td style="text-align: right;">0.507952</td>
<td style="text-align: right;">26.22</td>
<td style="text-align: right;">35.80</td>
<td style="text-align: right;">40.84</td>
<td style="text-align: right;">44.13</td>
<td style="text-align: right;">46.56</td>
<td style="text-align: right;">48.48</td>
<td style="text-align: right;">50.09</td>
<td style="text-align: right;">51.45</td>
<td style="text-align: right;">52.64</td>
<td style="text-align: right;">53.69</td>
<td style="text-align: right;">54.63</td>
<td style="text-align: right;">55.48</td>
<td style="text-align: right;">56.24</td>
<td style="text-align: right;">56.94</td>
<td style="text-align: right;">57.58</td>
<td style="text-align: right;">58.18</td>
</tr>
<tr>
<td>rl09_final</td>
<td style="text-align: right;">0.519173</td>
<td style="text-align: right;">36.36</td>
<td style="text-align: right;">42.15</td>
<td style="text-align: right;">45.09</td>
<td style="text-align: right;">46.98</td>
<td style="text-align: right;">48.34</td>
<td style="text-align: right;">49.39</td>
<td style="text-align: right;">50.24</td>
<td style="text-align: right;">50.94</td>
<td style="text-align: right;">51.54</td>
<td style="text-align: right;">52.06</td>
<td style="text-align: right;">52.52</td>
<td style="text-align: right;">52.93</td>
<td style="text-align: right;">53.31</td>
<td style="text-align: right;">53.65</td>
<td style="text-align: right;">53.97</td>
<td style="text-align: right;">54.26</td>
</tr>
<tr>
<td>rl10_final</td>
<td style="text-align: right;">0.519834</td>
<td style="text-align: right;">36.59</td>
<td style="text-align: right;">41.77</td>
<td style="text-align: right;">44.32</td>
<td style="text-align: right;">45.96</td>
<td style="text-align: right;">47.18</td>
<td style="text-align: right;">48.16</td>
<td style="text-align: right;">48.96</td>
<td style="text-align: right;">49.66</td>
<td style="text-align: right;">50.27</td>
<td style="text-align: right;">50.81</td>
<td style="text-align: right;">51.30</td>
<td style="text-align: right;">51.75</td>
<td style="text-align: right;">52.17</td>
<td style="text-align: right;">52.55</td>
<td style="text-align: right;">52.91</td>
<td style="text-align: right;">53.24</td>
</tr>
<tr>
<td>rl11_final</td>
<td style="text-align: right;">0.520187</td>
<td style="text-align: right;">36.80</td>
<td style="text-align: right;">43.68</td>
<td style="text-align: right;">47.17</td>
<td style="text-align: right;">49.47</td>
<td style="text-align: right;">51.18</td>
<td style="text-align: right;">52.53</td>
<td style="text-align: right;">53.67</td>
<td style="text-align: right;">54.64</td>
<td style="text-align: right;">55.49</td>
<td style="text-align: right;">56.25</td>
<td style="text-align: right;">56.94</td>
<td style="text-align: right;">57.56</td>
<td style="text-align: right;">58.13</td>
<td style="text-align: right;">58.66</td>
<td style="text-align: right;">59.14</td>
<td style="text-align: right;">59.59</td>
</tr>
<tr>
<td>rl12_final</td>
<td style="text-align: right;">0.518223</td>
<td style="text-align: right;">37.39</td>
<td style="text-align: right;">43.10</td>
<td style="text-align: right;">46.08</td>
<td style="text-align: right;">48.09</td>
<td style="text-align: right;">49.61</td>
<td style="text-align: right;">50.81</td>
<td style="text-align: right;">51.81</td>
<td style="text-align: right;">52.65</td>
<td style="text-align: right;">53.38</td>
<td style="text-align: right;">54.03</td>
<td style="text-align: right;">54.60</td>
<td style="text-align: right;">55.11</td>
<td style="text-align: right;">55.58</td>
<td style="text-align: right;">56.00</td>
<td style="text-align: right;">56.39</td>
<td style="text-align: right;">56.76</td>
</tr>
<tr>
<td>trace13_pt1_adam_ordered_pt_second</td>
<td style="text-align: right;">0.511257</td>
<td style="text-align: right;">33.56</td>
<td style="text-align: right;">40.14</td>
<td style="text-align: right;">43.49</td>
<td style="text-align: right;">45.71</td>
<td style="text-align: right;">47.34</td>
<td style="text-align: right;">48.61</td>
<td style="text-align: right;">49.63</td>
<td style="text-align: right;">50.49</td>
<td style="text-align: right;">51.23</td>
<td style="text-align: right;">51.88</td>
<td style="text-align: right;">52.45</td>
<td style="text-align: right;">52.97</td>
<td style="text-align: right;">53.44</td>
<td style="text-align: right;">53.88</td>
<td style="text-align: right;">54.29</td>
<td style="text-align: right;">54.66</td>
</tr>
<tr>
<td>trace14_pt1_adam_global_pt_second</td>
<td style="text-align: right;">0.510219</td>
<td style="text-align: right;">32.81</td>
<td style="text-align: right;">39.59</td>
<td style="text-align: right;">43.05</td>
<td style="text-align: right;">45.35</td>
<td style="text-align: right;">47.09</td>
<td style="text-align: right;">48.48</td>
<td style="text-align: right;">49.63</td>
<td style="text-align: right;">50.63</td>
<td style="text-align: right;">51.50</td>
<td style="text-align: right;">52.28</td>
<td style="text-align: right;">52.98</td>
<td style="text-align: right;">53.63</td>
<td style="text-align: right;">54.23</td>
<td style="text-align: right;">54.79</td>
<td style="text-align: right;">55.31</td>
<td style="text-align: right;">55.81</td>
</tr>
<tr>
<td>rl13_final</td>
<td style="text-align: right;">0.519089</td>
<td style="text-align: right;">37.82</td>
<td style="text-align: right;">43.15</td>
<td style="text-align: right;">45.85</td>
<td style="text-align: right;">47.64</td>
<td style="text-align: right;">48.96</td>
<td style="text-align: right;">50.01</td>
<td style="text-align: right;">50.88</td>
<td style="text-align: right;">51.62</td>
<td style="text-align: right;">52.29</td>
<td style="text-align: right;">52.88</td>
<td style="text-align: right;">53.42</td>
<td style="text-align: right;">53.92</td>
<td style="text-align: right;">54.38</td>
<td style="text-align: right;">54.82</td>
<td style="text-align: right;">55.22</td>
<td style="text-align: right;">55.61</td>
</tr>
<tr>
<td>rl14_final</td>
<td style="text-align: right;">0.519770</td>
<td style="text-align: right;">37.96</td>
<td style="text-align: right;">43.29</td>
<td style="text-align: right;">46.07</td>
<td style="text-align: right;">47.95</td>
<td style="text-align: right;">49.35</td>
<td style="text-align: right;">50.47</td>
<td style="text-align: right;">51.39</td>
<td style="text-align: right;">52.17</td>
<td style="text-align: right;">52.85</td>
<td style="text-align: right;">53.44</td>
<td style="text-align: right;">53.97</td>
<td style="text-align: right;">54.44</td>
<td style="text-align: right;">54.87</td>
<td style="text-align: right;">55.26</td>
<td style="text-align: right;">55.62</td>
<td style="text-align: right;">55.95</td>
</tr>
<tr>
<td>trace15_direct_rl_all_trace_ordered_continue_adam_pt_second</td>
<td style="text-align: right;">0.511704</td>
<td style="text-align: right;">34.08</td>
<td style="text-align: right;">40.40</td>
<td style="text-align: right;">43.58</td>
<td style="text-align: right;">45.72</td>
<td style="text-align: right;">47.33</td>
<td style="text-align: right;">48.64</td>
<td style="text-align: right;">49.73</td>
<td style="text-align: right;">50.67</td>
<td style="text-align: right;">51.50</td>
<td style="text-align: right;">52.25</td>
<td style="text-align: right;">52.93</td>
<td style="text-align: right;">53.55</td>
<td style="text-align: right;">54.12</td>
<td style="text-align: right;">54.65</td>
<td style="text-align: right;">55.15</td>
<td style="text-align: right;">55.61</td>
</tr>
<tr>
<td>trace16_direct_rl_all_trace_ordered_fresh_adam_pt_second</td>
<td style="text-align: right;">0.510470</td>
<td style="text-align: right;">34.04</td>
<td style="text-align: right;">40.55</td>
<td style="text-align: right;">43.81</td>
<td style="text-align: right;">45.99</td>
<td style="text-align: right;">47.63</td>
<td style="text-align: right;">48.94</td>
<td style="text-align: right;">50.03</td>
<td style="text-align: right;">50.95</td>
<td style="text-align: right;">51.75</td>
<td style="text-align: right;">52.46</td>
<td style="text-align: right;">53.08</td>
<td style="text-align: right;">53.63</td>
<td style="text-align: right;">54.14</td>
<td style="text-align: right;">54.60</td>
<td style="text-align: right;">55.02</td>
<td style="text-align: right;">55.41</td>
</tr>
<tr>
<td>rl15_final</td>
<td style="text-align: right;">0.521709</td>
<td style="text-align: right;">38.81</td>
<td style="text-align: right;">44.09</td>
<td style="text-align: right;">46.74</td>
<td style="text-align: right;">48.47</td>
<td style="text-align: right;">49.75</td>
<td style="text-align: right;">50.75</td>
<td style="text-align: right;">51.58</td>
<td style="text-align: right;">52.29</td>
<td style="text-align: right;">52.90</td>
<td style="text-align: right;">53.43</td>
<td style="text-align: right;">53.90</td>
<td style="text-align: right;">54.33</td>
<td style="text-align: right;">54.71</td>
<td style="text-align: right;">55.05</td>
<td style="text-align: right;">55.38</td>
<td style="text-align: right;">55.68</td>
</tr>
<tr>
<td>rl16_final</td>
<td style="text-align: right;">0.518890</td>
<td style="text-align: right;">38.04</td>
<td style="text-align: right;">43.33</td>
<td style="text-align: right;">46.02</td>
<td style="text-align: right;">47.82</td>
<td style="text-align: right;">49.17</td>
<td style="text-align: right;">50.25</td>
<td style="text-align: right;">51.15</td>
<td style="text-align: right;">51.93</td>
<td style="text-align: right;">52.62</td>
<td style="text-align: right;">53.23</td>
<td style="text-align: right;">53.78</td>
<td style="text-align: right;">54.28</td>
<td style="text-align: right;">54.74</td>
<td style="text-align: right;">55.17</td>
<td style="text-align: right;">55.57</td>
<td style="text-align: right;">55.95</td>
</tr>
</tbody>
</table></div></details>
<details><summary>Chess pretraining, 64 × 16: all evaluated checkpoints · full pass@1–16</summary><div class="interleave-table"><table>
<thead>
<tr>
<th>Checkpoint</th>
<th style="text-align: right;">PT loss ↓</th>
<th style="text-align: right;">Pass@1 (%)</th>
<th style="text-align: right;">Pass@2 (%)</th>
<th style="text-align: right;">Pass@3 (%)</th>
<th style="text-align: right;">Pass@4 (%)</th>
<th style="text-align: right;">Pass@5 (%)</th>
<th style="text-align: right;">Pass@6 (%)</th>
<th style="text-align: right;">Pass@7 (%)</th>
<th style="text-align: right;">Pass@8 (%)</th>
<th style="text-align: right;">Pass@9 (%)</th>
<th style="text-align: right;">Pass@10 (%)</th>
<th style="text-align: right;">Pass@11 (%)</th>
<th style="text-align: right;">Pass@12 (%)</th>
<th style="text-align: right;">Pass@13 (%)</th>
<th style="text-align: right;">Pass@14 (%)</th>
<th style="text-align: right;">Pass@15 (%)</th>
<th style="text-align: right;">Pass@16 (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>pt5b_sft3_full</td>
<td style="text-align: right;">0.512429</td>
<td style="text-align: right;">19.11</td>
<td style="text-align: right;">29.06</td>
<td style="text-align: right;">35.26</td>
<td style="text-align: right;">39.58</td>
<td style="text-align: right;">42.83</td>
<td style="text-align: right;">45.39</td>
<td style="text-align: right;">47.47</td>
<td style="text-align: right;">49.21</td>
<td style="text-align: right;">50.70</td>
<td style="text-align: right;">51.98</td>
<td style="text-align: right;">53.12</td>
<td style="text-align: right;">54.12</td>
<td style="text-align: right;">55.03</td>
<td style="text-align: right;">55.85</td>
<td style="text-align: right;">56.60</td>
<td style="text-align: right;">57.30</td>
</tr>
<tr>
<td>pt2p5b_sft3_first</td>
<td style="text-align: right;">0.525055</td>
<td style="text-align: right;">13.85</td>
<td style="text-align: right;">22.14</td>
<td style="text-align: right;">27.84</td>
<td style="text-align: right;">32.05</td>
<td style="text-align: right;">35.33</td>
<td style="text-align: right;">37.99</td>
<td style="text-align: right;">40.19</td>
<td style="text-align: right;">42.07</td>
<td style="text-align: right;">43.69</td>
<td style="text-align: right;">45.11</td>
<td style="text-align: right;">46.37</td>
<td style="text-align: right;">47.49</td>
<td style="text-align: right;">48.49</td>
<td style="text-align: right;">49.41</td>
<td style="text-align: right;">50.24</td>
<td style="text-align: right;">51.01</td>
</tr>
<tr>
<td>r64_e01_rl3000_full</td>
<td style="text-align: right;">0.534776</td>
<td style="text-align: right;">38.19</td>
<td style="text-align: right;">44.33</td>
<td style="text-align: right;">47.46</td>
<td style="text-align: right;">49.56</td>
<td style="text-align: right;">51.15</td>
<td style="text-align: right;">52.42</td>
<td style="text-align: right;">53.47</td>
<td style="text-align: right;">54.37</td>
<td style="text-align: right;">55.14</td>
<td style="text-align: right;">55.81</td>
<td style="text-align: right;">56.41</td>
<td style="text-align: right;">56.94</td>
<td style="text-align: right;">57.42</td>
<td style="text-align: right;">57.84</td>
<td style="text-align: right;">58.23</td>
<td style="text-align: right;">58.58</td>
</tr>
<tr>
<td>r64_e02_rl1_a</td>
<td style="text-align: right;">0.526597</td>
<td style="text-align: right;">36.20</td>
<td style="text-align: right;">43.11</td>
<td style="text-align: right;">46.70</td>
<td style="text-align: right;">49.11</td>
<td style="text-align: right;">50.92</td>
<td style="text-align: right;">52.36</td>
<td style="text-align: right;">53.55</td>
<td style="text-align: right;">54.55</td>
<td style="text-align: right;">55.43</td>
<td style="text-align: right;">56.20</td>
<td style="text-align: right;">56.89</td>
<td style="text-align: right;">57.52</td>
<td style="text-align: right;">58.09</td>
<td style="text-align: right;">58.63</td>
<td style="text-align: right;">59.13</td>
<td style="text-align: right;">59.59</td>
</tr>
<tr>
<td>r64_e02_trace_chronological</td>
<td style="text-align: right;">0.538013</td>
<td style="text-align: right;">36.79</td>
<td style="text-align: right;">43.46</td>
<td style="text-align: right;">46.86</td>
<td style="text-align: right;">49.12</td>
<td style="text-align: right;">50.81</td>
<td style="text-align: right;">52.15</td>
<td style="text-align: right;">53.26</td>
<td style="text-align: right;">54.20</td>
<td style="text-align: right;">55.01</td>
<td style="text-align: right;">55.73</td>
<td style="text-align: right;">56.36</td>
<td style="text-align: right;">56.94</td>
<td style="text-align: right;">57.46</td>
<td style="text-align: right;">57.94</td>
<td style="text-align: right;">58.38</td>
<td style="text-align: right;">58.78</td>
</tr>
<tr>
<td>r64_e02_rl2_b</td>
<td style="text-align: right;">0.542875</td>
<td style="text-align: right;">38.63</td>
<td style="text-align: right;">44.73</td>
<td style="text-align: right;">47.76</td>
<td style="text-align: right;">49.75</td>
<td style="text-align: right;">51.24</td>
<td style="text-align: right;">52.44</td>
<td style="text-align: right;">53.44</td>
<td style="text-align: right;">54.30</td>
<td style="text-align: right;">55.06</td>
<td style="text-align: right;">55.74</td>
<td style="text-align: right;">56.34</td>
<td style="text-align: right;">56.88</td>
<td style="text-align: right;">57.37</td>
<td style="text-align: right;">57.82</td>
<td style="text-align: right;">58.22</td>
<td style="text-align: right;">58.58</td>
</tr>
<tr>
<td>r64_e03_rl1_full</td>
<td style="text-align: right;">0.524071</td>
<td style="text-align: right;">36.24</td>
<td style="text-align: right;">43.72</td>
<td style="text-align: right;">47.51</td>
<td style="text-align: right;">50.00</td>
<td style="text-align: right;">51.85</td>
<td style="text-align: right;">53.31</td>
<td style="text-align: right;">54.52</td>
<td style="text-align: right;">55.56</td>
<td style="text-align: right;">56.46</td>
<td style="text-align: right;">57.26</td>
<td style="text-align: right;">57.98</td>
<td style="text-align: right;">58.63</td>
<td style="text-align: right;">59.22</td>
<td style="text-align: right;">59.77</td>
<td style="text-align: right;">60.27</td>
<td style="text-align: right;">60.74</td>
</tr>
<tr>
<td>r64_e03_trace_chronological</td>
<td style="text-align: right;">0.535673</td>
<td style="text-align: right;">36.59</td>
<td style="text-align: right;">43.79</td>
<td style="text-align: right;">47.35</td>
<td style="text-align: right;">49.64</td>
<td style="text-align: right;">51.32</td>
<td style="text-align: right;">52.63</td>
<td style="text-align: right;">53.70</td>
<td style="text-align: right;">54.61</td>
<td style="text-align: right;">55.40</td>
<td style="text-align: right;">56.09</td>
<td style="text-align: right;">56.71</td>
<td style="text-align: right;">57.27</td>
<td style="text-align: right;">57.77</td>
<td style="text-align: right;">58.24</td>
<td style="text-align: right;">58.66</td>
<td style="text-align: right;">59.05</td>
</tr>
<tr>
<td>r64_e03_rl2_full</td>
<td style="text-align: right;">0.543711</td>
<td style="text-align: right;">38.20</td>
<td style="text-align: right;">44.32</td>
<td style="text-align: right;">47.40</td>
<td style="text-align: right;">49.43</td>
<td style="text-align: right;">50.93</td>
<td style="text-align: right;">52.10</td>
<td style="text-align: right;">53.06</td>
<td style="text-align: right;">53.87</td>
<td style="text-align: right;">54.57</td>
<td style="text-align: right;">55.18</td>
<td style="text-align: right;">55.72</td>
<td style="text-align: right;">56.21</td>
<td style="text-align: right;">56.65</td>
<td style="text-align: right;">57.06</td>
<td style="text-align: right;">57.43</td>
<td style="text-align: right;">57.77</td>
</tr>
<tr>
<td>r64_e04_rl1_a</td>
<td style="text-align: right;">0.536907</td>
<td style="text-align: right;">28.75</td>
<td style="text-align: right;">35.52</td>
<td style="text-align: right;">39.26</td>
<td style="text-align: right;">41.83</td>
<td style="text-align: right;">43.76</td>
<td style="text-align: right;">45.30</td>
<td style="text-align: right;">46.57</td>
<td style="text-align: right;">47.66</td>
<td style="text-align: right;">48.61</td>
<td style="text-align: right;">49.45</td>
<td style="text-align: right;">50.21</td>
<td style="text-align: right;">50.90</td>
<td style="text-align: right;">51.53</td>
<td style="text-align: right;">52.12</td>
<td style="text-align: right;">52.66</td>
<td style="text-align: right;">53.18</td>
</tr>
<tr>
<td>r64_e04_pt2_trace_mixed</td>
<td style="text-align: right;">0.509955</td>
<td style="text-align: right;">29.47</td>
<td style="text-align: right;">38.18</td>
<td style="text-align: right;">42.81</td>
<td style="text-align: right;">45.90</td>
<td style="text-align: right;">48.20</td>
<td style="text-align: right;">50.03</td>
<td style="text-align: right;">51.53</td>
<td style="text-align: right;">52.80</td>
<td style="text-align: right;">53.90</td>
<td style="text-align: right;">54.87</td>
<td style="text-align: right;">55.74</td>
<td style="text-align: right;">56.52</td>
<td style="text-align: right;">57.23</td>
<td style="text-align: right;">57.89</td>
<td style="text-align: right;">58.49</td>
<td style="text-align: right;">59.05</td>
</tr>
<tr>
<td>r64_e04_rl2_b</td>
<td style="text-align: right;">0.522745</td>
<td style="text-align: right;">36.16</td>
<td style="text-align: right;">42.52</td>
<td style="text-align: right;">45.77</td>
<td style="text-align: right;">47.91</td>
<td style="text-align: right;">49.49</td>
<td style="text-align: right;">50.74</td>
<td style="text-align: right;">51.77</td>
<td style="text-align: right;">52.64</td>
<td style="text-align: right;">53.39</td>
<td style="text-align: right;">54.06</td>
<td style="text-align: right;">54.65</td>
<td style="text-align: right;">55.19</td>
<td style="text-align: right;">55.68</td>
<td style="text-align: right;">56.14</td>
<td style="text-align: right;">56.56</td>
<td style="text-align: right;">56.96</td>
</tr>
<tr>
<td>r64_e05_rl1_full</td>
<td style="text-align: right;">0.536972</td>
<td style="text-align: right;">29.17</td>
<td style="text-align: right;">36.15</td>
<td style="text-align: right;">39.87</td>
<td style="text-align: right;">42.36</td>
<td style="text-align: right;">44.21</td>
<td style="text-align: right;">45.66</td>
<td style="text-align: right;">46.85</td>
<td style="text-align: right;">47.84</td>
<td style="text-align: right;">48.68</td>
<td style="text-align: right;">49.42</td>
<td style="text-align: right;">50.06</td>
<td style="text-align: right;">50.63</td>
<td style="text-align: right;">51.15</td>
<td style="text-align: right;">51.61</td>
<td style="text-align: right;">52.04</td>
<td style="text-align: right;">52.43</td>
</tr>
<tr>
<td>r64_e05_pt2_trace_mixed</td>
<td style="text-align: right;">0.509888</td>
<td style="text-align: right;">30.24</td>
<td style="text-align: right;">39.10</td>
<td style="text-align: right;">43.74</td>
<td style="text-align: right;">46.80</td>
<td style="text-align: right;">49.04</td>
<td style="text-align: right;">50.79</td>
<td style="text-align: right;">52.21</td>
<td style="text-align: right;">53.40</td>
<td style="text-align: right;">54.42</td>
<td style="text-align: right;">55.31</td>
<td style="text-align: right;">56.11</td>
<td style="text-align: right;">56.82</td>
<td style="text-align: right;">57.47</td>
<td style="text-align: right;">58.06</td>
<td style="text-align: right;">58.61</td>
<td style="text-align: right;">59.12</td>
</tr>
<tr>
<td>r64_e05_rl2_full</td>
<td style="text-align: right;">0.520700</td>
<td style="text-align: right;">36.09</td>
<td style="text-align: right;">42.56</td>
<td style="text-align: right;">45.99</td>
<td style="text-align: right;">48.32</td>
<td style="text-align: right;">50.09</td>
<td style="text-align: right;">51.51</td>
<td style="text-align: right;">52.71</td>
<td style="text-align: right;">53.74</td>
<td style="text-align: right;">54.65</td>
<td style="text-align: right;">55.47</td>
<td style="text-align: right;">56.20</td>
<td style="text-align: right;">56.87</td>
<td style="text-align: right;">57.49</td>
<td style="text-align: right;">58.05</td>
<td style="text-align: right;">58.57</td>
<td style="text-align: right;">59.05</td>
</tr>
</tbody>
</table></div></details>



<details><summary>OLMo2 math: all schedules with completed evaluations</summary><div class="interleave-table"><table><thead><tr><th>Checkpoint</th><th>PT loss ↓</th><th>Numina response loss ↓</th><th>SkyEasy pass@1 (%)</th><th>SkyEasy pass@2 (%)</th><th>SkyEasy pass@3 (%)</th><th>SkyEasy pass@4 (%)</th><th>SkyEasy pass@5 (%)</th><th>SkyEasy pass@6 (%)</th><th>SkyEasy pass@7 (%)</th><th>SkyEasy pass@8 (%)</th><th>SkyEasy pass@9 (%)</th><th>SkyEasy pass@10 (%)</th><th>SkyEasy pass@11 (%)</th><th>SkyEasy pass@12 (%)</th><th>SkyEasy pass@13 (%)</th><th>SkyEasy pass@14 (%)</th><th>SkyEasy pass@15 (%)</th><th>SkyEasy pass@16 (%)</th><th>GSM8K greedy (%)</th><th>MATH 5,000 greedy (%)</th></tr></thead><tbody><tr><td>Shared NTP A (22.5B tokens)</td><td>1.319161</td><td>0.640536</td><td>5.36</td><td>9.11</td><td>12.06</td><td>14.52</td><td>16.65</td><td>18.53</td><td>20.21</td><td>21.73</td><td>23.12</td><td>24.41</td><td>25.61</td><td>26.73</td><td>27.78</td><td>28.77</td><td>29.71</td><td>30.60</td><td>11.22</td><td>15.58</td></tr><tr><td>Baseline · end of NTP B (45B tokens)</td><td>1.227608</td><td>0.578258</td><td>9.28</td><td>15.08</td><td>19.39</td><td>22.82</td><td>25.66</td><td>28.08</td><td>30.19</td><td>32.05</td><td>33.72</td><td>35.23</td><td>36.61</td><td>37.89</td><td>39.07</td><td>40.17</td><td>41.21</td><td>42.20</td><td>21.00</td><td>23.78</td></tr><tr><td>Baseline · final RL, update 1,500 of 3,000</td><td>1.235369</td><td>0.601326</td><td>26.20</td><td>35.39</td><td>40.78</td><td>44.56</td><td>47.43</td><td>49.73</td><td>51.62</td><td>53.23</td><td>54.61</td><td>55.83</td><td>56.91</td><td>57.88</td><td>58.76</td><td>59.57</td><td>60.31</td><td>61.00</td><td>36.16</td><td>29.68</td></tr><tr><td>Baseline · final RL, update 3,000</td><td>1.240630</td><td>0.616258</td><td>28.94</td><td>38.03</td><td>43.26</td><td>46.90</td><td>49.70</td><td>51.98</td><td>53.90</td><td>55.55</td><td>57.01</td><td>58.31</td><td>59.49</td><td>60.57</td><td>61.57</td><td>62.50</td><td>63.38</td><td>64.20</td><td>42.53</td><td>31.34</td></tr><tr><td>Shared first RL · full, fresh AdamW</td><td>1.329831</td><td>0.676669</td><td>16.76</td><td>24.13</td><td>28.87</td><td>32.32</td><td>35.00</td><td>37.18</td><td>39.01</td><td>40.58</td><td>41.97</td><td>43.21</td><td>44.33</td><td>45.36</td><td>46.31</td><td>47.20</td><td>48.02</td><td>48.80</td><td>24.18</td><td>21.08</td></tr><tr><td>Direct continuation, fresh AdamW · end of NTP B</td><td>1.235861</td><td>0.582820</td><td>10.25</td><td>16.50</td><td>20.98</td><td>24.45</td><td>27.28</td><td>29.66</td><td>31.71</td><td>33.52</td><td>35.13</td><td>36.58</td><td>37.90</td><td>39.12</td><td>40.25</td><td>41.29</td><td>42.27</td><td>43.20</td><td>22.37</td><td>24.38</td></tr><tr><td>Direct continuation, fresh AdamW · final RL</td><td>1.242338</td><td>0.603771</td><td>24.36</td><td>33.66</td><td>39.41</td><td>43.57</td><td>46.78</td><td>49.38</td><td>51.54</td><td>53.37</td><td>54.96</td><td>56.35</td><td>57.59</td><td>58.70</td><td>59.70</td><td>60.61</td><td>61.44</td><td>62.20</td><td>35.63</td><td>28.52</td></tr><tr><td>First RL · full, AdamW state kept</td><td>1.327666</td><td>0.670230</td><td>16.44</td><td>23.54</td><td>28.16</td><td>31.64</td><td>34.44</td><td>36.76</td><td>38.75</td><td>40.49</td><td>42.03</td><td>43.41</td><td>44.65</td><td>45.78</td><td>46.81</td><td>47.75</td><td>48.61</td><td>49.40</td><td>22.59</td><td>21.08</td></tr><tr><td>Direct continuation, AdamW state kept · end of NTP B</td><td>1.235521</td><td>0.583451</td><td>10.39</td><td>16.53</td><td>21.00</td><td>24.55</td><td>27.50</td><td>30.03</td><td>32.25</td><td>34.22</td><td>36.00</td><td>37.62</td><td>39.11</td><td>40.49</td><td>41.78</td><td>42.99</td><td>44.12</td><td>45.20</td><td>21.83</td><td>23.70</td></tr><tr><td>Direct continuation, AdamW state kept · final RL</td><td>1.241096</td><td>0.603600</td><td>24.52</td><td>33.39</td><td>38.71</td><td>42.45</td><td>45.31</td><td>47.60</td><td>49.52</td><td>51.16</td><td>52.59</td><td>53.85</td><td>54.98</td><td>55.99</td><td>56.91</td><td>57.74</td><td>58.50</td><td>59.20</td><td>33.66</td><td>28.98</td></tr><tr><td>PT + traces · full/full · end of NTP B</td><td>1.234848</td><td>0.583262</td><td>16.83</td><td>24.99</td><td>30.24</td><td>34.09</td><td>37.10</td><td>39.57</td><td>41.65</td><td>43.44</td><td>45.02</td><td>46.42</td><td>47.69</td><td>48.84</td><td>49.89</td><td>50.86</td><td>51.76</td><td>52.60</td><td>28.35</td><td>25.38</td></tr><tr><td>PT + traces · full/full · final RL</td><td>1.238009</td><td>0.591252</td><td>24.89</td><td>34.39</td><td>39.92</td><td>43.76</td><td>46.68</td><td>49.04</td><td>51.01</td><td>52.72</td><td>54.22</td><td>55.57</td><td>56.78</td><td>57.90</td><td>58.92</td><td>59.87</td><td>60.76</td><td>61.60</td><td>31.31</td><td>28.98</td></tr><tr><td>PT + traces · A/B · end of NTP B (final RL pending)</td><td>1.234661</td><td>0.582597</td><td>17.79</td><td>25.80</td><td>31.04</td><td>34.93</td><td>38.02</td><td>40.57</td><td>42.73</td><td>44.60</td><td>46.24</td><td>47.70</td><td>49.02</td><td>50.22</td><td>51.32</td><td>52.34</td><td>53.30</td><td>54.20</td><td>27.07</td><td>25.56</td></tr><tr><td>Trace-only · A/B · first RL on A, from the 45B baseline</td><td>1.235791</td><td>0.604160</td><td>25.79</td><td>34.90</td><td>40.26</td><td>44.05</td><td>46.96</td><td>49.29</td><td>51.22</td><td>52.84</td><td>54.24</td><td>55.46</td><td>56.53</td><td>57.48</td><td>58.33</td><td>59.09</td><td>59.77</td><td>60.40</td><td>29.87</td><td>30.20</td></tr><tr><td>Trace-only · A/B · after trace training</td><td>1.245500</td><td>0.628006</td><td>29.15</td><td>38.27</td><td>43.37</td><td>46.92</td><td>49.64</td><td>51.82</td><td>53.65</td><td>55.20</td><td>56.55</td><td>57.73</td><td>58.79</td><td>59.74</td><td>60.61</td><td>61.40</td><td>62.12</td><td>62.80</td><td>33.36</td><td>30.82</td></tr><tr><td>Trace-only · A/B · final RL on B</td><td>1.251273</td><td>0.646142</td><td>31.82</td><td>40.96</td><td>46.12</td><td>49.69</td><td>52.38</td><td>54.50</td><td>56.25</td><td>57.73</td><td>59.00</td><td>60.11</td><td>61.10</td><td>61.99</td><td>62.79</td><td>63.52</td><td>64.19</td><td>64.80</td><td>37.38</td><td>32.60</td></tr></tbody></table></div></details>

<details><summary>Chess SFT: all thirteen checkpoints, full pass@1–16</summary><div class="interleave-table"><table><thead><tr><th>Checkpoint</th><th>PT loss ↓</th><th>Pass@1 (%)</th><th>Pass@2 (%)</th><th>Pass@3 (%)</th><th>Pass@4 (%)</th><th>Pass@5 (%)</th><th>Pass@6 (%)</th><th>Pass@7 (%)</th><th>Pass@8 (%)</th><th>Pass@9 (%)</th><th>Pass@10 (%)</th><th>Pass@11 (%)</th><th>Pass@12 (%)</th><th>Pass@13 (%)</th><th>Pass@14 (%)</th><th>Pass@15 (%)</th><th>Pass@16 (%)</th><th>Strict format (%)</th></tr></thead><tbody><tr><td>Shared SFT A</td><td>0.672537</td><td>2.81</td><td>5.24</td><td>7.36</td><td>9.22</td><td>10.87</td><td>12.35</td><td>13.68</td><td>14.89</td><td>16.00</td><td>17.02</td><td>17.97</td><td>18.85</td><td>19.68</td><td>20.46</td><td>21.20</td><td>21.89</td><td>60.20</td></tr><tr><td>Baseline · SFT B</td><td>0.722182</td><td>11.36</td><td>19.02</td><td>24.50</td><td>28.64</td><td>31.90</td><td>34.56</td><td>36.78</td><td>38.69</td><td>40.34</td><td>41.81</td><td>43.12</td><td>44.31</td><td>45.39</td><td>46.39</td><td>47.31</td><td>48.18</td><td>91.08</td></tr><tr><td>Shared first RL · A</td><td>0.763008</td><td>19.35</td><td>24.87</td><td>27.85</td><td>29.86</td><td>31.38</td><td>32.60</td><td>33.64</td><td>34.54</td><td>35.34</td><td>36.06</td><td>36.71</td><td>37.30</td><td>37.85</td><td>38.35</td><td>38.82</td><td>39.26</td><td>3.85</td></tr><tr><td>Shared first RL · full</td><td>0.730352</td><td>25.50</td><td>31.93</td><td>35.47</td><td>37.92</td><td>39.80</td><td>41.31</td><td>42.57</td><td>43.66</td><td>44.60</td><td>45.42</td><td>46.16</td><td>46.82</td><td>47.42</td><td>47.96</td><td>48.46</td><td>48.92</td><td>1.88</td></tr><tr><td>Baseline · final RL</td><td>0.760608</td><td>33.54</td><td>40.09</td><td>43.53</td><td>45.83</td><td>47.55</td><td>48.93</td><td>50.08</td><td>51.06</td><td>51.92</td><td>52.69</td><td>53.37</td><td>53.99</td><td>54.55</td><td>55.07</td><td>55.56</td><td>56.01</td><td>4.93</td></tr><tr><td>Direct continuation · A/B · SFT B</td><td>0.728390</td><td>11.66</td><td>19.58</td><td>25.30</td><td>29.64</td><td>33.06</td><td>35.83</td><td>38.15</td><td>40.12</td><td>41.83</td><td>43.33</td><td>44.66</td><td>45.87</td><td>46.96</td><td>47.96</td><td>48.88</td><td>49.73</td><td>91.23</td></tr><tr><td>Direct continuation · A/B · final RL</td><td>0.817552</td><td>31.63</td><td>39.11</td><td>43.11</td><td>45.75</td><td>47.70</td><td>49.24</td><td>50.50</td><td>51.56</td><td>52.47</td><td>53.28</td><td>53.99</td><td>54.63</td><td>55.22</td><td>55.75</td><td>56.24</td><td>56.69</td><td>0.99</td></tr><tr><td>SFT + traces · A/B · SFT B</td><td>0.816657</td><td>22.29</td><td>29.10</td><td>32.83</td><td>35.35</td><td>37.25</td><td>38.76</td><td>40.03</td><td>41.12</td><td>42.09</td><td>42.95</td><td>43.73</td><td>44.44</td><td>45.10</td><td>45.72</td><td>46.29</td><td>46.82</td><td>18.84</td></tr><tr><td>SFT + traces · A/B · final RL</td><td>0.794882</td><td>32.66</td><td>39.16</td><td>42.60</td><td>44.93</td><td>46.67</td><td>48.06</td><td>49.20</td><td>50.17</td><td>51.00</td><td>51.72</td><td>52.37</td><td>52.94</td><td>53.45</td><td>53.92</td><td>54.34</td><td>54.73</td><td>0.35</td></tr><tr><td>Direct continuation · full/full · SFT B</td><td>0.727651</td><td>11.60</td><td>19.38</td><td>24.97</td><td>29.24</td><td>32.64</td><td>35.44</td><td>37.80</td><td>39.85</td><td>41.63</td><td>43.22</td><td>44.65</td><td>45.95</td><td>47.13</td><td>48.21</td><td>49.21</td><td>50.14</td><td>91.27</td></tr><tr><td>Direct continuation · full/full · final RL</td><td>0.778875</td><td>31.11</td><td>38.80</td><td>42.96</td><td>45.70</td><td>47.70</td><td>49.27</td><td>50.56</td><td>51.66</td><td>52.61</td><td>53.45</td><td>54.21</td><td>54.89</td><td>55.52</td><td>56.11</td><td>56.65</td><td>57.16</td><td>1.34</td></tr><tr><td>SFT + traces · full/full · SFT B</td><td>0.796639</td><td>26.74</td><td>34.66</td><td>38.98</td><td>41.93</td><td>44.18</td><td>45.99</td><td>47.49</td><td>48.78</td><td>49.90</td><td>50.89</td><td>51.78</td><td>52.60</td><td>53.35</td><td>54.05</td><td>54.71</td><td>55.34</td><td>17.15</td></tr><tr><td>SFT + traces · full/full · final RL</td><td>0.792758</td><td>32.97</td><td>39.47</td><td>42.92</td><td>45.24</td><td>46.98</td><td>48.35</td><td>49.47</td><td>50.42</td><td>51.23</td><td>51.95</td><td>52.59</td><td>53.16</td><td>53.68</td><td>54.16</td><td>54.59</td><td>55.00</td><td>0.47</td></tr></tbody></table></div></details>

<details><summary>Llama SFT: all thirteen checkpoints, validation loss and greedy accuracy</summary><div class="interleave-table"><table><thead><tr><th>Checkpoint</th><th>Numina SFT loss ↓</th><th>GSM8K (%)</th><th>MATH-500 (%)</th><th>Polaris raw (%)</th><th>Polaris valid-reference (%)</th></tr></thead><tbody><tr><td>Shared SFT A</td><td>0.4770</td><td>65.66</td><td>33.20</td><td>7.81</td><td>8.15</td></tr><tr><td>Baseline · complete SFT</td><td>0.4455</td><td>70.58</td><td>39.00</td><td>7.62</td><td>7.94</td></tr><tr><td>Shared first RL · A</td><td>0.6346</td><td>68.31</td><td>31.80</td><td>14.45</td><td>15.07</td></tr><tr><td>Shared first RL · full</td><td>0.6161</td><td>70.51</td><td>30.60</td><td>13.96</td><td>14.56</td></tr><tr><td>Baseline · final RL</td><td>0.7440</td><td>76.27</td><td>37.20</td><td>16.70</td><td>17.41</td></tr><tr><td>Direct continuation · A/B · SFT B</td><td>0.4441</td><td>69.52</td><td>38.80</td><td>9.86</td><td>10.29</td></tr><tr><td>Direct continuation · A/B · final RL</td><td>0.6246</td><td>74.83</td><td>40.00</td><td>16.11</td><td>16.80</td></tr><tr><td>SFT + traces · A/B · SFT B</td><td>0.4534</td><td>71.49</td><td>39.60</td><td>15.92</td><td>16.60</td></tr><tr><td>SFT + traces · A/B · final RL</td><td>0.5363</td><td>72.86</td><td>40.60</td><td>17.29</td><td>18.02</td></tr><tr><td>Direct continuation · full/full · SFT B</td><td>0.4440</td><td>70.43</td><td>39.40</td><td>9.38</td><td>9.78</td></tr><tr><td>Direct continuation · full/full · final RL</td><td>0.6218</td><td>74.07</td><td>39.60</td><td>15.62</td><td>16.29</td></tr><tr><td>SFT + traces · full/full · SFT B</td><td>0.4520</td><td>69.60</td><td>38.40</td><td>15.23</td><td>15.89</td></tr><tr><td>SFT + traces · full/full · final RL</td><td>0.5427</td><td>72.25</td><td>39.40</td><td>15.23</td><td>15.89</td></tr></tbody></table></div></details>

<h2 id="data-and-results">Data and results</h2>

The files below contain the loss, accuracy, and paired comparisons used in this post. Each SFT study has five training schedules and thirteen evaluated checkpoints, counting shared checkpoints once.

- [Chess PT checkpoint metrics, both studies](/assets/blog/interleaving-pt-rl/chess-results.json)
- [Chess PT paired comparisons and intervals](/assets/blog/interleaving-pt-rl/chess-comparisons.json)
- [OLMo2 math checkpoint metrics](/assets/blog/interleaving-pt-rl/math-results.json)
- [Chess SFT: all thirteen checkpoints](/assets/blog/interleaving-pt-rl/chess-sft-results.json)
- [Llama SFT: all thirteen checkpoints](/assets/blog/interleaving-pt-rl/numina-sft-results.json)
