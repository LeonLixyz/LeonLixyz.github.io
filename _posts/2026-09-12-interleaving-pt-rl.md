---
layout: distill
published: false
title: "Beyond Pretraining Followed by Post-Training: An Exploration of Interleaving NTP and RL"
description: Can earlier RL help a model learn from the data it sees next? We test this in five chess and math studies, before and after final RL.
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
  - name: "Why bring RL earlier?"
  - name: "What we compare"
  - name: "Chess: interleaving pretraining and RL"
  - name: "Math: interleaving pretraining and RL"
  - name: "Chess: interleaving SFT and RL"
  - name: "Llama: interleaving SFT and RL"
  - name: "What did we learn?"
  - name: "Evaluation and limits"
  - name: "Complete results"
  - name: "Data and results"
bibliography: interleaving-pt-rl.bib
_styles: |
  d-article .interleave-table { overflow-x: auto; max-width: 100%; margin: 1.5rem 0; }
  d-article .interleave-table table { width: 100%; font-size: 12px; line-height: 1.55; font-variant-numeric: tabular-nums; margin: 0; }
  d-article .interleave-table th, d-article .interleave-table td { padding: 10px 12px; vertical-align: top; }
  d-article .interleave-table th { white-space: nowrap; }
  d-article .interleave-table td:first-child { min-width: 190px; }
  d-article .interleave-table td:not(:first-child) { white-space: nowrap; }
  d-article .interleave-figure img { display: block; width: 100%; height: auto; background: white; }
  d-article .interleave-figure figcaption { margin-top: 0.8rem; }
  d-article details { margin-bottom: 1rem; }
  d-article details summary { cursor: pointer; font-weight: 600; }
  d-article details .interleave-table { margin-bottom: 0; }
  d-article h2 { scroll-margin-top: 80px; }
  d-article .interleave-text-table td { white-space: normal !important; text-align: left !important; }
  d-article mjx-container[display="true"] { overflow-x: auto; overflow-y: hidden; max-width: 100%; padding: 0.7rem 0; }
---

People learn from both demonstrations and trial and error, often alternating between the two. We might study a worked example, try a problem ourselves, and then return to the example with a better sense of what we missed. Our attempts can shape what we learn from the explanations and examples that follow.

Language-model training, on the other hand, commonly follows a fixed sequence: pretraining, supervised fine-tuning (SFT), and then reinforcement learning (RL). Pretraining and SFT use next-token prediction (NTP) to learn from provided sequences, including worked solutions. RL then trains on the model’s sampled responses using reward feedback. In this sequence, the model finishes most of its learning from provided data before RL begins.

This raises a question about the order of training: **can earlier RL help a model learn from the data it sees next?** Both NTP and RL can improve problem-solving performance. We ask whether progress through RL also helps subsequent NTP training, and whether alternating the two produces a better final model.

We test this by inserting RL before pretraining or SFT is finished. The model then returns to NTP on the remaining data, sometimes mixed with successful responses generated during RL. We call these responses **traces**. A final RL stage completes each run.

Across five studies in chess and math, **interleaving improves some intermediate checkpoints but does not consistently improve the final model**. Traces often raise intermediate pass@1, but the baseline usually catches up or finishes ahead after final RL. There are also positive final results: in Llama 3.2 3B, interleaving improves MATH-500 accuracy and Numina loss, while reducing GSM8K accuracy. We find no clear evidence that earlier RL makes subsequent NTP learning more efficient.

<h2 id="why-bring-rl-earlier">Why bring RL earlier?</h2>

Our earlier work found that pretraining loss predicts post-RL performance in the settings we studied, and that models pretrained for longer improve faster under RL. Here we ask about the reverse direction: can an RL-trained model learn better during further NTP training? [Understanding Reasoning from Pretraining to Post-Training](https://arxiv.org/abs/2607.16097)

A model does not need to finish pretraining before it can solve useful tasks. RL on tasks it can already solve might develop strategies or representations that help it learn from later examples. The question is whether that benefit extends beyond the tasks used for RL.

The two objectives can also pull the model in different directions. For a dataset $D$, NTP minimizes the average loss over supervised tokens:

$$
\mathcal{L}_{\mathrm{NTP}}(\theta;D)
= -\frac{1}{N_D}\sum_{x\in D}\sum_t
m_t(x)\log p_\theta(x_t\mid x_{\lt t}),
\qquad N_D=\sum_{x\in D}\sum_t m_t(x).
$$

Here $\theta$ denotes the model parameters. The mask $m_t(x)$ is 1 when token $t$ contributes to the loss and 0 otherwise; $N_D$ counts those tokens. Pretraining uses text tokens as targets. SFT uses solution tokens, with prompts and any environment replies masked out. We use the same calculation on held-out data to measure loss.

RL aims to increase the expected reward of generated responses:

$$
J(\theta)
=\mathbb{E}_{q\sim Q,\;y\sim p_\theta(\cdot\mid q)}[r(q,y)].
$$

Here $Q$ is the training prompt distribution, $y$ is a sampled response, and $r(q,y)$ is its reward. This describes the task objective; the policy optimizer uses a separate update loss. Higher reward need not mean lower NTP loss. RL may favor successful responses that differ from the offline targets, and returning to NTP may undo some of those changes.

We test two ways to use what RL produces: continue training from its weights, or train on its successful traces. With traces, we can also return to the earlier supervised checkpoint and transfer progress through the data alone.

Recent work uses related forms of alternating training:

- **MAI-Thinking-1** trains midtrained checkpoints on RL rollouts and then resumes RL. This helps recover from unstable runs and transfer progress to newer base checkpoints. It also mixes midtraining data with traces to preserve long-context behavior. [Report, §3.1.4](https://microsoft.ai/pdf/mai-thinking-1.pdf#page=34)
- **Ring-Zero** selects and shortens successful RL trajectories, fine-tunes the original base model on them, and continues RL. The supervised stage aims to improve stability and control response length. [Paper, §3.1.2](https://arxiv.org/html/2607.12395v2#S3.SS1.SSS2)
- **ReMiT** uses an RL-tuned reference model to change the weights on token losses during midtraining. It reports improvements in base-model evaluations and subsequent post-training. [Paper, §3](https://arxiv.org/html/2602.03075v1#S3)

These methods use RL in different ways. Our experiments test simpler approaches: ordinary NTP continuation and training on successful traces. Success with the methods above does not guarantee that these simpler approaches will help with unfinished pretraining.

<h2 id="what-we-compare">What we compare</h2>

We ask two questions:

1. **Does earlier RL help the next NTP stage?** After training on the remaining data, we compare held-out loss and task performance with a model that has received no RL.
2. **Does it improve the final model?** After all RL updates, we compare against a baseline that finishes supervised training before starting RL.

At each stage, the differences from baseline are

$$
\begin{aligned}
\Delta\mathcal{L}_{\mathrm{NTP}}
&=\mathcal{L}_{\mathrm{NTP}}(\theta_{\mathrm{interleaved}};D_{\mathrm{heldout}})
-\mathcal{L}_{\mathrm{NTP}}(\theta_{\mathrm{baseline}};D_{\mathrm{heldout}}),\\
\Delta P_k
&=\operatorname{pass@}k(\theta_{\mathrm{interleaved}})
-\operatorname{pass@}k(\theta_{\mathrm{baseline}}).
\end{aligned}
$$

Negative $\Delta\mathcal{L}_{\mathrm{NTP}}$ means lower loss; positive $\Delta P_k$ means higher task success. Accuracy differences are in percentage points. We compare both before and after final RL because an intermediate gain may not survive the rest of training.

NTP A and NTP B are the two parts of the supervised training. In the pretraining (PT) studies, they include pretraining data and the usual task-supervision mixture. In the SFT studies, they contain worked solutions and start from a fixed pretrained model.

<div class="interleave-table interleave-text-table" markdown="1">

| Route                      |                             Training sequence |                       What carries forward from RL |
| -------------------------- | --------------------------------------------: | -------------------------------------------------: |
| Baseline                   |                            NTP A → NTP B → RL |                                 No intermediate RL |
| Direct continuation        |                     NTP A → RL₁ → NTP B → RL₂ |                                        RL₁ weights |
| Trace mixing               | NTP A → RL₁ → NTP B + successful traces → RL₂ | Generated data; initialization specified per study |
| Trace-only (earlier chess) |      Complete PT → RL₁ → trace training → RL₂ |           Generated data; no remaining ordinary PT |

</div>

Training on traces can start from either the supervised checkpoint or the RL checkpoint. That choice matters. In the two SFT studies, direct continuation keeps the RL weights, while training on traces starts again from SFT A. We specify the starting checkpoint for each study below.

The baseline receives 3,000 RL updates at the end. Interleaved runs receive 1,500 in the middle and 1,500 at the end. Before final RL, the interleaved model has therefore had 1,500 RL updates that the baseline has not. These intermediate comparisons do not use equal compute. Training on traces also adds tokens and updates, and some runs differ in optimizer resets and learning-rate schedules. **We match the stated data and RL update budgets, not total FLOPs.**

<div class="interleave-table interleave-text-table" markdown="1">

| Study            |                            Model |           Supervised budget | RL prompts × samples |
| ---------------- | -------------------------------: | --------------------------: | -------------------: |
| Earlier chess PT |                              47M | 5B PT tokens + ordinary SFT |              256 × 8 |
| Later chess PT   |                              47M | 5B PT tokens + ordinary SFT |              64 × 16 |
| Math PT          |                         OLMo2 1B |         45B PT tokens + CoT |              64 × 16 |
| Chess SFT        | 47M; fixed 9.18B-token PT parent |         77,717 SFT examples |              64 × 16 |
| Math SFT         |                     Llama 3.2 3B | 823,505 Numina SFT examples |              64 × 16 |

</div>

The two chess PT studies use **256 × 8** and **64 × 16** prompts × samples per RL update. We compare models within each study. Other differences in checkpoints, data, and evaluation mean that we cannot attribute differences between studies to the number of prompts or samples alone.

We report cross-entropy, also called log loss, where lower is better. For task performance, we report sampled pass@k when multiple responses are available and label greedy evaluations separately. Pass@1 measures success in one sampled attempt; pass@k measures success within k attempts. The estimator is given below. Lower loss, higher pass@1, and higher pass@16 need not occur together.

<h2 id="chess-interleaving-pretraining-and-rl">Chess: interleaving pretraining and RL</h2>

<h3>First study: what survives a return to pretraining?</h3>

We began by testing direct PT continuation, trace-only training, PT mixed with traces, different trace orders, and different ways to handle optimizer state. We also split PT into two schedules without inserting RL. This control lets us check whether a change comes from inserting RL or simply from splitting the PT schedule.

The direct-continuation run trains on 2.5B PT tokens, runs RL, and then trains on the remaining 2.5B PT tokens with fresh AdamW and no RL traces. Before final RL:

<div class="interleave-table" markdown="1">

| Checkpoint                   | PT loss ↓ | Pass@1 (%) | Pass@2 (%) | Pass@4 (%) | Pass@8 (%) | Pass@16 (%) |
| ---------------------------- | --------: | ---------: | ---------: | ---------: | ---------: | ----------: |
| After first RL               |  0.540289 |      30.79 |      36.39 |      41.26 |      45.60 |       49.53 |
| After direct PT continuation |  0.507826 |      17.85 |      27.68 |      38.27 |      47.86 |       55.68 |
| One 5B PT schedule; no RL    |  0.512429 |      19.11 |      29.06 |      39.58 |      49.21 |       57.30 |
| Two 2.5B PT schedules; no RL |  0.507432 |      18.23 |      28.09 |      38.71 |      48.62 |       57.43 |

</div>

Returning to PT lowers pass@1 from **30.79% to 17.85%**, but improves loss and raises pass@16 from **49.53% to 55.68%**. The model becomes less likely to succeed in one attempt, yet more likely to succeed within sixteen attempts.

<figure class="l-body interleave-figure"><img src="/assets/blog/interleaving-pt-rl/direct-continuation.svg" loading="lazy" alt="Pass at k after the first RL stage, after direct PT continuation, and after uninterrupted PT."><figcaption>Figure 1. Returning to PT reduces pass@1 but raises pass@16 relative to the RL checkpoint. Uninterrupted PT still has higher pass@k than direct continuation at every displayed k.</figcaption></figure>

The no-RL controls give us a stricter comparison. Direct continuation has lower loss than uninterrupted 5B PT, but slightly higher loss than split PT without RL: **0.507826 versus 0.507432**. It also has lower pass@k than both controls at every displayed k. This comparison gives no evidence that RL improves the next PT stage.

Traces give higher intermediate pass@1. Trace-only training restarted from PT weights reaches **37.70%**; PT plus traces restarted from PT weights reaches **33.50%**. Continuing from RL weights with PT and all successful traces reaches **34.04%** with fresh AdamW. These runs differ in their data and optimization, so the differences cannot be assigned to one component alone.

After final RL, baseline reaches **37.86% pass@1 and 54.80% pass@16**. Trace-only training reaches **38.49% / 53.85%**; direct continuation reaches **36.36% / 54.26%**; and RL weights followed by PT and all traces with fresh AdamW reaches **38.01% / 54.59%**.

Across all fifteen alternatives, the confidence intervals for final pass@1 and pass@16 differences include zero after adjustment for thirty comparisons. We cannot establish a final gain, though this does not prove that the runs are equivalent. The appendix includes all sixteen experiments.

<h3>Second study: does the choice of RL data matter?</h3>

The second study uses 64 prompts × 16 samples. We test trace-only training and PT mixed with traces; both continue from RL weights with fresh AdamW. We compare **full/full**, where both RL stages use the same 53,156 prompts, and **A/B**, where they use disjoint sets of 26,578 prompts each.

The full/full PT-plus-trace stage improves all three metrics relative to its RL checkpoint: PT loss **0.536972 → 0.509888**, pass@1 **29.17% → 30.24%**, and pass@16 **52.43% → 59.12%**. It also beats uninterrupted 5B PT on those metrics. This is a positive intermediate result, though extra trace training and schedule differences prevent us from attributing it to the RL weights alone.

After final RL:

<div class="interleave-table" markdown="1">

| Checkpoint              | PT loss ↓ | Pass@1 (%) | Pass@2 (%) | Pass@4 (%) | Pass@8 (%) | Pass@16 (%) |
| ----------------------- | --------: | ---------: | ---------: | ---------: | ---------: | ----------: |
| Baseline                |  0.534776 |      38.19 |      44.33 |      49.56 |      54.37 |       58.58 |
| Trace-only · A/B        |  0.542875 |      38.63 |      44.73 |      49.75 |      54.30 |       58.58 |
| Trace-only · full/full  |  0.543711 |      38.20 |      44.32 |      49.43 |      53.87 |       57.77 |
| PT + traces · A/B       |  0.522745 |      36.16 |      42.52 |      47.91 |      52.64 |       56.96 |
| PT + traces · full/full |  0.520700 |      36.09 |      42.56 |      48.32 |      53.74 |       59.05 |

</div>

<figure class="l-body interleave-figure"><img src="/assets/blog/interleaving-pt-rl/later-chess-final.svg" loading="lazy" alt="Final PT loss, pass at 1, and pass at 16 in the five later chess PT experiments."><figcaption>Figure 2. PT plus traces gives lower final PT loss but also lower pass@1. Dashed lines show baseline performance; dots show the measured results.</figcaption></figure>

For full/full PT plus traces, final pass@1 is **2.10 percentage points below baseline**, with an adjusted paired-bootstrap interval of **[−3.78, −0.39]**. Pass@16 is **0.47 points above baseline**, with an interval of **[−2.40, +3.24]**. These intervals support a pass@1 decrease but do not establish a pass@16 increase. Neither data schedule gives a consistent final advantage.

<h2 id="math-interleaving-pretraining-and-rl">Math: interleaving pretraining and RL</h2>

We next test OLMo2 1B with a 45B-token PT budget, followed by SkyEasy RL. We compare baseline, PT plus traces, and direct continuation. PT plus traces starts again from the first PT checkpoint; direct continuation keeps the first RL checkpoint. These runs use fresh AdamW at stage boundaries.

<div class="interleave-table" markdown="1">

| Checkpoint                            | PT loss ↓ | Pass@1 (%) | Pass@2 (%) | Pass@4 (%) | Pass@8 (%) | Pass@16 (%) |
| ------------------------------------- | --------: | ---------: | ---------: | ---------: | ---------: | ----------: |
| Baseline · before final RL            |  1.227608 |       9.28 |      15.08 |      22.82 |      32.05 |       42.20 |
| PT + traces · before final RL         |  1.234848 |      16.83 |      24.99 |      34.09 |      43.44 |       52.60 |
| Direct continuation · before final RL |  1.235861 |      10.25 |      16.50 |      24.45 |      33.52 |       43.20 |
| Baseline · after final RL             |  1.240630 |      28.94 |      38.03 |      46.90 |      55.55 |       64.20 |
| PT + traces · after final RL          |  1.238009 |      24.89 |      34.39 |      43.76 |      52.72 |       61.60 |
| Direct continuation · after final RL  |  1.242338 |      24.36 |      33.66 |      43.57 |      53.37 |       62.20 |

</div>

PT plus traces raises intermediate pass@1 from **9.28% to 16.83%**, and pass@16 from **42.20% to 52.60%**, relative to baseline. Both interleaved runs have slightly higher PT loss, however. They improve task performance without improving the pretraining loss.

After final RL, baseline has higher accuracy at every displayed SkyEasy k. It also leads on the greedy benchmarks: **42.53% GSM8K / 31.34% MATH**, compared with **31.31% / 28.98%** for PT plus traces and **35.63% / 28.52%** for direct continuation. Here MATH is the 5,000-question test set; the later Llama study uses MATH-500.

As in chess, higher intermediate accuracy does not lead to higher final accuracy. This study has no matched control that splits PT without RL, so it cannot separate the effect of RL from the effect of splitting training.

<h2 id="chess-interleaving-sft-and-rl">Chess: interleaving SFT and RL</h2>

We then ask whether interleaving helps when the remaining supervised data consists entirely of task solutions. We hold pretraining fixed and insert RL into SFT. This tests whether RL helps further fine-tuning.

All five runs start from the same 47M PT-only model, trained on **9.181735B PT target tokens**. We split **77,717 SFT examples** into groups A and B with no shared puzzle identities. RL uses 64 × 16 samples and either disjoint A/B prompt sets or the full prompt set. Each new SFT and RL stage uses fresh AdamW.

The runs share SFT A, then follow these schedules:

<div class="interleave-table interleave-text-table" markdown="1">

| Approach                        |                          Remaining path after SFT A | SFT B initialization |
| ------------------------------- | --------------------------------------------------: | -------------------: |
| Baseline                        |                               SFT B → full RL 3,000 |                SFT A |
| Direct continuation · A/B       |                     RL A 1,500 → SFT B → RL B 1,500 |                 RL A |
| SFT + traces · A/B              |          RL A 1,500 → SFT B + A traces → RL B 1,500 |       Reset to SFT A |
| Direct continuation · full/full |               Full RL 1,500 → SFT B → full RL 1,500 |        First full RL |
| SFT + traces · full/full        | Full RL 1,500 → SFT B + full traces → full RL 1,500 |       Reset to SFT A |

</div>

The trace runs use extra supervised updates and carry information from RL through generated data. Direct continuation carries it through the model weights.

<h3>After the second SFT stage</h3>

After SFT B, before final RL:

<div class="interleave-table" markdown="1">

| Approach                        | PT loss ↓ | Pass@1 (%) | Pass@4 (%) | Pass@16 (%) |
| ------------------------------- | --------: | ---------: | ---------: | ----------: |
| Baseline                        |    0.7222 |      11.36 |      28.64 |       48.18 |
| Direct continuation · A/B       |    0.7284 |      11.66 |      29.64 |       49.73 |
| SFT + traces · A/B              |    0.8167 |      22.29 |      35.35 |       46.82 |
| Direct continuation · full/full |    0.7277 |      11.60 |      29.24 |       50.14 |
| SFT + traces · full/full        |    0.7966 |      26.74 |      41.93 |       55.34 |

</div>

Direct continuation gives slightly higher pass@1 than baseline: **+0.29 points for A/B** and **+0.24 points for full/full**. Training on traces gives much larger gains: **+10.93** and **+15.38 points**.

Comparing with the first RL checkpoint shows what survives SFT. Full-data RL reaches **25.50% pass@1**. Ordinary SFT B then reduces it to **11.60%**, close to the SFT baseline. Training on traces instead reaches **26.74%**, even though it starts again from SFT-A weights. Successful traces can therefore transfer high single-attempt accuracy without retaining the RL weights.

The loss column measures **held-out pretraining text**, not SFT solutions. All four interleaved runs have higher loss than the SFT baseline. Their higher task accuracy comes with worse predictions on the pretraining data.

<h3>After final RL</h3>

<div class="interleave-table" markdown="1">

| Approach                        | PT loss ↓ | Pass@1 (%) | Pass@2 (%) | Pass@4 (%) | Pass@8 (%) | Pass@16 (%) |
| ------------------------------- | --------: | ---------: | ---------: | ---------: | ---------: | ----------: |
| Baseline                        |    0.7606 |      33.54 |      40.09 |      45.83 |      51.06 |       56.01 |
| Direct continuation · A/B       |    0.8176 |      31.63 |      39.11 |      45.75 |      51.56 |       56.69 |
| SFT + traces · A/B              |    0.7949 |      32.66 |      39.16 |      44.93 |      50.17 |       54.73 |
| Direct continuation · full/full |    0.7789 |      31.11 |      38.80 |      45.70 |      51.66 |       57.16 |
| SFT + traces · full/full        |    0.7928 |      32.97 |      39.47 |      45.24 |      50.42 |       55.00 |

</div>

<figure class="l-body interleave-figure"><img src="/assets/blog/interleaving-pt-rl/chess-sft-stages.svg" loading="lazy" alt="Chess pass at 1 and pass at 16 before and after final RL. Large intermediate trace gains do not become a final pass at 1 advantage."><figcaption>Figure 3. The trace runs enter final RL with much higher pass@1 than baseline but finish below it. Baseline receives 3,000 final RL updates; interleaved runs receive 1,500 after 1,500 earlier. Lines connect the two evaluations, not a measured learning curve.</figcaption></figure>

Baseline finishes with the highest pass@1 and lowest PT loss. Direct full/full continuation has the highest pass@16, **1.15 points above baseline**, but lower pass@1. Both trace runs finish below baseline at pass@1 and pass@16. These are results from single runs; we have not established their statistical significance.

All interleaved models improve during final RL. But once both schedules finish their RL budgets, none retains an advantage over baseline at pass@1.

<h2 id="llama-interleaving-sft-and-rl">Llama: interleaving SFT and RL</h2>

We use the same five schedules with **Llama 3.2 3B**, **823,505 NuminaMath-CoT SFT examples**, and **51,996 Polaris RL prompts**. SFT A and B contain 411,687 and 411,818 examples. RL A and B each contain 25,998 prompts. The context length is 8,192 tokens, and RL uses 64 × 16 samples.

As in chess, direct continuation keeps the RL weights, while trace training starts again from SFT A. One optimizer detail differs: baseline completes SFT A and B with continuous Adam state and one full-epoch learning-rate schedule. Interleaved runs use fresh AdamW at the transitions. Small loss differences may therefore reflect these choices as well as the inserted RL.

Before final RL, baseline Numina loss is **0.445460**. Direct continuation gives slightly lower loss: **0.444113** with A/B and **0.443960** with full/full. The trace runs give slightly higher loss: **0.453373** and **0.451987**. The lower losses are encouraging, but the optimizer differences prevent a clear claim that RL makes NTP learning more efficient.

The A/B trace model also improves intermediate task performance. It reaches **71.49% GSM8K, 39.60% MATH-500, and 15.92% Polaris**, compared with baseline’s **70.58%, 39.00%, and 7.62%**. At this point, it has already received additional RL and trace training.

After final RL:

<div class="interleave-table" markdown="1">

| Approach                        | Numina SFT loss ↓ | GSM8K (%) | MATH-500 (%) | Polaris (%) |
| ------------------------------- | ----------------: | --------: | -----------: | ----------: |
| Baseline                        |            0.7440 |     76.27 |        37.20 |       16.70 |
| Direct continuation · A/B       |            0.6246 |     74.83 |        40.00 |       16.11 |
| SFT + traces · A/B              |            0.5363 |     72.86 |        40.60 |       17.29 |
| Direct continuation · full/full |            0.6218 |     74.07 |        39.60 |       15.62 |
| SFT + traces · full/full        |            0.5427 |     72.25 |        39.40 |       15.23 |

</div>

<figure class="l-body interleave-figure"><img src="/assets/blog/interleaving-pt-rl/llama-final-differences.svg" loading="lazy" alt="Final Llama accuracy differences from baseline on GSM8K, MATH-500, and Polaris. Every interleaved run improves MATH-500 and reduces GSM8K; only A/B trace training improves Polaris."><figcaption>Figure 4. Final greedy accuracy relative to baseline. All four interleaved runs improve MATH-500 and reduce GSM8K. The figure shows measured differences without uncertainty intervals.</figcaption></figure>

Here, interleaving gives some final gains. All four runs have **lower Numina loss and higher MATH-500 accuracy**, but **lower GSM8K accuracy**. A/B trace training gives the best MATH-500 and Polaris scores: **+3.40 and +0.59 points** over baseline, with **−3.41 points on GSM8K**. No interleaved run improves all three benchmarks.

The loss result also depends on when we measure it. A/B trace training has slightly _higher_ loss than baseline before final RL, but much _lower_ loss afterward: **0.5363 versus 0.7440**. This shows that the final model predicts the supervised data better. It does not show that the earlier NTP stage learned more efficiently. The starting model and the placement of RL updates can also affect how much final RL changes predictions on that data.

Polaris scores include all 1,024 held-out questions. There are 42 invalid references, counted as failures for every model. A/B trace training answers 177/1,024 correctly, compared with baseline’s 171/1,024. We have not tested this six-question difference for significance. All three benchmarks use greedy decoding; this study has no sampled pass@2–16 results.

<h2 id="what-did-we-learn">What did we learn?</h2>

**We did not find a consistent final advantage from moving RL earlier.** There are useful intermediate gains and some final gains, but three distinctions matter when interpreting them.

**Learning from traces can help without retaining RL weights.** Training on successful traces raises intermediate accuracy, even when it starts from the checkpoint before RL. This shows that generated data can transfer useful behavior. It does not tell us whether RL-trained weights help the model learn from the remaining offline data.

**A better intermediate model may not be a better final model.** The chess trace runs can start final RL well ahead at pass@1 and still finish below baseline. Before that final stage, baseline has received no RL. We need the final comparison to know whether moving part of the RL budget earlier helps the full training schedule.

**The answer depends on what we measure.** Lower NTP loss can come with lower task accuracy, and higher pass@16 can come with lower pass@1. In Llama, MATH-500 improves while GSM8K gets worse. These are specific gains and tradeoffs, rather than a general improvement.

To show that RL makes NTP learning more efficient, we would need to compare held-out loss throughout the next NTP stage: $\mathcal{L}_{\mathrm{heldout}}(u)$, where $u$ is the number of NTP tokens or the compute used after RL. That comparison needs the same data and controlled optimizer and learning-rate settings. A claim about total compute must also count intermediate RL and trace generation. Our before-and-after measurements do not answer that question fully.

These results apply to the schedules we tested. We insert RL halfway through PT or SFT and return to NTP once; trace-only runs add supervised replay after PT is complete. We do not test repeated alternation, adaptive task selection, or RL-guided token weighting. The result is a limitation of these simple continuation and trace-training approaches, not evidence that all forms of interleaving must fail.

<h2 id="evaluation-and-limits">Evaluation and limits</h2>

**Chess.** We evaluate on 1,480 held-out puzzles with sixteen samples per puzzle, temperature 1, and top-p 1. The context limit is 2,048 tokens, with at most 512 prompt tokens and 1,536 response tokens. Four puzzles from the original 1,484 exceed the prompt limit and are excluded. PT cross-entropy uses 8,388,608 held-out target tokens. The SFT study uses a separate PT model and a corrected rotary positional embedding (RoPE) configuration, so its results should be compared with its own baseline. We record compliance with the reasoning-tag format separately from move correctness.

For $N$ evaluation problems, let $c_i$ be the number of correct responses among $n=16$ samples for problem $i$. We estimate pass@k as

$$
\widehat{\operatorname{pass@}k}
=\frac{1}{N}\sum_{i=1}^{N}
\left[1-\frac{\binom{n-c_i}{k}}{\binom{n}{k}}\right],
\qquad 1\le k\le n,
$$

with $\binom{n-c_i}{k}=0$ when $n-c_i<k$. The ratio is the probability that a set of $k$ responses drawn without replacement from the samples contains no correct answer. Pass@1 is the mean sampled accuracy. Pass@16 is the fraction of problems with at least one correct answer among all sixteen samples. Neither directly measures response diversity.

**OLMo2 math.** SkyEasy uses 500 questions with sixteen samples each, temperature 1, and top-p 1, at the model’s native context length of 4,096. GSM8K and MATH use greedy decoding on 1,319 and 5,000 questions. PT loss combines the PT evaluation components. We also report Numina response loss on 100 questions, with prompts masked out.

**Llama math.** We use the model’s native chat format and greedy decoding with an 8,192-token context. GSM8K has 1,319 questions, MATH-500 has 500, and Polaris has 1,024. Numina loss averages over 423,311 assistant tokens from 1,036 records. This measures SFT solution loss; the OLMo2 and chess PT losses use different data. The download includes Polaris accuracy both with and without invalid references. Losses and benchmark variants should not be compared directly across studies.

**Uncertainty.** For the earlier chess studies, final comparisons use 20,000 paired bootstrap replicates, resampling puzzle identities. We adjust the intervals for thirty comparisons in the first study and eight in the second. These intervals describe uncertainty across puzzles for the trained models and sampled responses. They do not measure variation across training seeds. We do not report corresponding intervals for the SFT studies, math comparisons, or loss measurements, so small differences remain uncertain.

**Training budgets.** Within the intended comparisons, we match total RL updates and the stated original-data budgets. Total FLOPs can differ because of trace generation and training, optimizer resets, learning-rate schedules, and RL data allocation. A/B runs also change which prompts each RL stage sees. These experiments compare whole training schedules; they do not isolate every change in those schedules.

<h2 id="complete-results">Complete results</h2>

The first five tables report the two chess pretraining studies, including A/B variants and alternative final-RL datasets. The main text compares final checkpoints with the matching baseline in each study. Tables show rounded values; the data files keep full precision.

<details><summary>Earlier chess: all 16 experiments, before and after final RL</summary><div class="interleave-table"><table>
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
<td>RL weights → PT2, carry AdamW · after final RL</td>
<td style="text-align: right;">0.520721</td>
<td style="text-align: right;">36.90</td>
<td style="text-align: right;">42.27</td>
<td style="text-align: right;">47.10</td>
<td style="text-align: right;">51.50</td>
<td style="text-align: right;">55.20</td>
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
<td>RL weights → PT2 + late replay, carry AdamW · after final RL</td>
<td style="text-align: right;">0.519272</td>
<td style="text-align: right;">36.93</td>
<td style="text-align: right;">41.69</td>
<td style="text-align: right;">45.71</td>
<td style="text-align: right;">49.34</td>
<td style="text-align: right;">52.84</td>
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
<td>PT weights + PT AdamW → PT2 + traces, ordered · after final RL</td>
<td style="text-align: right;">0.520507</td>
<td style="text-align: right;">37.53</td>
<td style="text-align: right;">42.41</td>
<td style="text-align: right;">46.49</td>
<td style="text-align: right;">49.79</td>
<td style="text-align: right;">52.43</td>
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
<td>PT weights + PT AdamW → PT2 + traces, shuffled · after final RL</td>
<td style="text-align: right;">0.518676</td>
<td style="text-align: right;">37.11</td>
<td style="text-align: right;">42.04</td>
<td style="text-align: right;">46.13</td>
<td style="text-align: right;">49.61</td>
<td style="text-align: right;">52.77</td>
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
<td>RL weights + RL AdamW → PT2 + all traces · after final RL</td>
<td style="text-align: right;">0.521367</td>
<td style="text-align: right;">37.62</td>
<td style="text-align: right;">42.58</td>
<td style="text-align: right;">46.74</td>
<td style="text-align: right;">50.36</td>
<td style="text-align: right;">53.51</td>
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
<tr>
<td>RL weights + fresh AdamW → PT2 + all traces · after final RL</td>
<td style="text-align: right;">0.520661</td>
<td style="text-align: right;">38.01</td>
<td style="text-align: right;">42.94</td>
<td style="text-align: right;">47.04</td>
<td style="text-align: right;">50.84</td>
<td style="text-align: right;">54.59</td>
</tr>
</tbody>
</table></div></details>
<details><summary>Later chess: all five experiments, before and after final RL</summary><div class="interleave-table"><table>
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
<details><summary>Earlier chess: six additional final endpoints on the larger RL set</summary><p>These runs use 53,225 RL training prompts. The main comparison uses 28,419 prompts, so we report these results separately.</p><div class="interleave-table"><table>
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
<tr>
<td>rl11_final_prompts28419</td>
<td style="text-align: right;">0.520721</td>
<td style="text-align: right;">36.90</td>
<td style="text-align: right;">42.27</td>
<td style="text-align: right;">45.15</td>
<td style="text-align: right;">47.10</td>
<td style="text-align: right;">48.57</td>
<td style="text-align: right;">49.74</td>
<td style="text-align: right;">50.69</td>
<td style="text-align: right;">51.50</td>
<td style="text-align: right;">52.18</td>
<td style="text-align: right;">52.78</td>
<td style="text-align: right;">53.31</td>
<td style="text-align: right;">53.77</td>
<td style="text-align: right;">54.19</td>
<td style="text-align: right;">54.56</td>
<td style="text-align: right;">54.90</td>
<td style="text-align: right;">55.20</td>
</tr>
<tr>
<td>rl12_final_prompts28419</td>
<td style="text-align: right;">0.519272</td>
<td style="text-align: right;">36.93</td>
<td style="text-align: right;">41.69</td>
<td style="text-align: right;">44.10</td>
<td style="text-align: right;">45.71</td>
<td style="text-align: right;">46.91</td>
<td style="text-align: right;">47.86</td>
<td style="text-align: right;">48.66</td>
<td style="text-align: right;">49.34</td>
<td style="text-align: right;">49.94</td>
<td style="text-align: right;">50.48</td>
<td style="text-align: right;">50.96</td>
<td style="text-align: right;">51.41</td>
<td style="text-align: right;">51.81</td>
<td style="text-align: right;">52.18</td>
<td style="text-align: right;">52.52</td>
<td style="text-align: right;">52.84</td>
</tr>
<tr>
<td>rl13_final_prompts28419</td>
<td style="text-align: right;">0.520507</td>
<td style="text-align: right;">37.53</td>
<td style="text-align: right;">42.41</td>
<td style="text-align: right;">44.89</td>
<td style="text-align: right;">46.49</td>
<td style="text-align: right;">47.64</td>
<td style="text-align: right;">48.51</td>
<td style="text-align: right;">49.21</td>
<td style="text-align: right;">49.79</td>
<td style="text-align: right;">50.28</td>
<td style="text-align: right;">50.70</td>
<td style="text-align: right;">51.07</td>
<td style="text-align: right;">51.40</td>
<td style="text-align: right;">51.70</td>
<td style="text-align: right;">51.97</td>
<td style="text-align: right;">52.21</td>
<td style="text-align: right;">52.43</td>
</tr>
<tr>
<td>rl14_final_prompts28419</td>
<td style="text-align: right;">0.518676</td>
<td style="text-align: right;">37.11</td>
<td style="text-align: right;">42.04</td>
<td style="text-align: right;">44.52</td>
<td style="text-align: right;">46.13</td>
<td style="text-align: right;">47.30</td>
<td style="text-align: right;">48.22</td>
<td style="text-align: right;">48.97</td>
<td style="text-align: right;">49.61</td>
<td style="text-align: right;">50.16</td>
<td style="text-align: right;">50.65</td>
<td style="text-align: right;">51.09</td>
<td style="text-align: right;">51.49</td>
<td style="text-align: right;">51.85</td>
<td style="text-align: right;">52.18</td>
<td style="text-align: right;">52.49</td>
<td style="text-align: right;">52.77</td>
</tr>
<tr>
<td>rl15_final_prompts28419</td>
<td style="text-align: right;">0.521367</td>
<td style="text-align: right;">37.62</td>
<td style="text-align: right;">42.58</td>
<td style="text-align: right;">45.09</td>
<td style="text-align: right;">46.74</td>
<td style="text-align: right;">47.96</td>
<td style="text-align: right;">48.91</td>
<td style="text-align: right;">49.70</td>
<td style="text-align: right;">50.36</td>
<td style="text-align: right;">50.94</td>
<td style="text-align: right;">51.44</td>
<td style="text-align: right;">51.88</td>
<td style="text-align: right;">52.28</td>
<td style="text-align: right;">52.63</td>
<td style="text-align: right;">52.95</td>
<td style="text-align: right;">53.25</td>
<td style="text-align: right;">53.51</td>
</tr>
<tr>
<td>rl16_final_prompts28419</td>
<td style="text-align: right;">0.520661</td>
<td style="text-align: right;">38.01</td>
<td style="text-align: right;">42.94</td>
<td style="text-align: right;">45.40</td>
<td style="text-align: right;">47.04</td>
<td style="text-align: right;">48.28</td>
<td style="text-align: right;">49.28</td>
<td style="text-align: right;">50.11</td>
<td style="text-align: right;">50.84</td>
<td style="text-align: right;">51.47</td>
<td style="text-align: right;">52.04</td>
<td style="text-align: right;">52.56</td>
<td style="text-align: right;">53.03</td>
<td style="text-align: right;">53.47</td>
<td style="text-align: right;">53.87</td>
<td style="text-align: right;">54.25</td>
<td style="text-align: right;">54.59</td>
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

<details><summary>OLMo2 math: selected three approaches, before and after final RL</summary><div class="interleave-table"><table><thead><tr><th>Checkpoint</th><th>PT loss ↓</th><th>Numina response loss ↓</th><th>SkyEasy pass@1 (%)</th><th>SkyEasy pass@2 (%)</th><th>SkyEasy pass@3 (%)</th><th>SkyEasy pass@4 (%)</th><th>SkyEasy pass@5 (%)</th><th>SkyEasy pass@6 (%)</th><th>SkyEasy pass@7 (%)</th><th>SkyEasy pass@8 (%)</th><th>SkyEasy pass@9 (%)</th><th>SkyEasy pass@10 (%)</th><th>SkyEasy pass@11 (%)</th><th>SkyEasy pass@12 (%)</th><th>SkyEasy pass@13 (%)</th><th>SkyEasy pass@14 (%)</th><th>SkyEasy pass@15 (%)</th><th>SkyEasy pass@16 (%)</th><th>GSM8K greedy (%)</th><th>MATH 5,000 greedy (%)</th></tr></thead><tbody><tr><td>Baseline · before final RL</td><td>1.227608</td><td>0.578258</td><td>9.28</td><td>15.08</td><td>19.39</td><td>22.82</td><td>25.66</td><td>28.08</td><td>30.19</td><td>32.05</td><td>33.72</td><td>35.23</td><td>36.61</td><td>37.89</td><td>39.07</td><td>40.17</td><td>41.21</td><td>42.20</td><td>21.00</td><td>23.78</td></tr><tr><td>PT + traces · before final RL</td><td>1.234848</td><td>0.583262</td><td>16.83</td><td>24.99</td><td>30.24</td><td>34.09</td><td>37.10</td><td>39.57</td><td>41.65</td><td>43.44</td><td>45.02</td><td>46.42</td><td>47.69</td><td>48.84</td><td>49.89</td><td>50.86</td><td>51.76</td><td>52.60</td><td>28.35</td><td>25.38</td></tr><tr><td>Direct continuation · before final RL</td><td>1.235861</td><td>0.582820</td><td>10.25</td><td>16.50</td><td>20.98</td><td>24.45</td><td>27.28</td><td>29.66</td><td>31.71</td><td>33.52</td><td>35.13</td><td>36.58</td><td>37.90</td><td>39.12</td><td>40.25</td><td>41.29</td><td>42.27</td><td>43.20</td><td>22.37</td><td>24.38</td></tr><tr><td>Baseline · after final RL</td><td>1.240630</td><td>0.616258</td><td>28.94</td><td>38.03</td><td>43.26</td><td>46.90</td><td>49.70</td><td>51.98</td><td>53.90</td><td>55.55</td><td>57.01</td><td>58.31</td><td>59.49</td><td>60.57</td><td>61.57</td><td>62.50</td><td>63.38</td><td>64.20</td><td>42.53</td><td>31.34</td></tr><tr><td>PT + traces · after final RL</td><td>1.238009</td><td>0.591252</td><td>24.89</td><td>34.39</td><td>39.92</td><td>43.76</td><td>46.68</td><td>49.04</td><td>51.01</td><td>52.72</td><td>54.22</td><td>55.57</td><td>56.78</td><td>57.90</td><td>58.92</td><td>59.87</td><td>60.76</td><td>61.60</td><td>31.31</td><td>28.98</td></tr><tr><td>Direct continuation · after final RL</td><td>1.242338</td><td>0.603771</td><td>24.36</td><td>33.66</td><td>39.41</td><td>43.57</td><td>46.78</td><td>49.38</td><td>51.54</td><td>53.37</td><td>54.96</td><td>56.35</td><td>57.59</td><td>58.70</td><td>59.70</td><td>60.61</td><td>61.44</td><td>62.20</td><td>35.63</td><td>28.52</td></tr></tbody></table></div></details>

<details><summary>Chess SFT–RL: all thirteen checkpoints, full pass@1–16</summary><div class="interleave-table"><table><thead><tr><th>Checkpoint</th><th>PT loss ↓</th><th>Pass@1 (%)</th><th>Pass@2 (%)</th><th>Pass@3 (%)</th><th>Pass@4 (%)</th><th>Pass@5 (%)</th><th>Pass@6 (%)</th><th>Pass@7 (%)</th><th>Pass@8 (%)</th><th>Pass@9 (%)</th><th>Pass@10 (%)</th><th>Pass@11 (%)</th><th>Pass@12 (%)</th><th>Pass@13 (%)</th><th>Pass@14 (%)</th><th>Pass@15 (%)</th><th>Pass@16 (%)</th><th>Strict format (%)</th></tr></thead><tbody><tr><td>Shared SFT A</td><td>0.672537</td><td>2.81</td><td>5.24</td><td>7.36</td><td>9.22</td><td>10.87</td><td>12.35</td><td>13.68</td><td>14.89</td><td>16.00</td><td>17.02</td><td>17.97</td><td>18.85</td><td>19.68</td><td>20.46</td><td>21.20</td><td>21.89</td><td>60.20</td></tr><tr><td>Baseline · SFT B</td><td>0.722182</td><td>11.36</td><td>19.02</td><td>24.50</td><td>28.64</td><td>31.90</td><td>34.56</td><td>36.78</td><td>38.69</td><td>40.34</td><td>41.81</td><td>43.12</td><td>44.31</td><td>45.39</td><td>46.39</td><td>47.31</td><td>48.18</td><td>91.08</td></tr><tr><td>Shared first RL · A</td><td>0.763008</td><td>19.35</td><td>24.87</td><td>27.85</td><td>29.86</td><td>31.38</td><td>32.60</td><td>33.64</td><td>34.54</td><td>35.34</td><td>36.06</td><td>36.71</td><td>37.30</td><td>37.85</td><td>38.35</td><td>38.82</td><td>39.26</td><td>3.85</td></tr><tr><td>Shared first RL · full</td><td>0.730352</td><td>25.50</td><td>31.93</td><td>35.47</td><td>37.92</td><td>39.80</td><td>41.31</td><td>42.57</td><td>43.66</td><td>44.60</td><td>45.42</td><td>46.16</td><td>46.82</td><td>47.42</td><td>47.96</td><td>48.46</td><td>48.92</td><td>1.88</td></tr><tr><td>Baseline · final RL</td><td>0.760608</td><td>33.54</td><td>40.09</td><td>43.53</td><td>45.83</td><td>47.55</td><td>48.93</td><td>50.08</td><td>51.06</td><td>51.92</td><td>52.69</td><td>53.37</td><td>53.99</td><td>54.55</td><td>55.07</td><td>55.56</td><td>56.01</td><td>4.93</td></tr><tr><td>Direct continuation · A/B · SFT B</td><td>0.728390</td><td>11.66</td><td>19.58</td><td>25.30</td><td>29.64</td><td>33.06</td><td>35.83</td><td>38.15</td><td>40.12</td><td>41.83</td><td>43.33</td><td>44.66</td><td>45.87</td><td>46.96</td><td>47.96</td><td>48.88</td><td>49.73</td><td>91.23</td></tr><tr><td>Direct continuation · A/B · final RL</td><td>0.817552</td><td>31.63</td><td>39.11</td><td>43.11</td><td>45.75</td><td>47.70</td><td>49.24</td><td>50.50</td><td>51.56</td><td>52.47</td><td>53.28</td><td>53.99</td><td>54.63</td><td>55.22</td><td>55.75</td><td>56.24</td><td>56.69</td><td>0.99</td></tr><tr><td>SFT + traces · A/B · SFT B</td><td>0.816657</td><td>22.29</td><td>29.10</td><td>32.83</td><td>35.35</td><td>37.25</td><td>38.76</td><td>40.03</td><td>41.12</td><td>42.09</td><td>42.95</td><td>43.73</td><td>44.44</td><td>45.10</td><td>45.72</td><td>46.29</td><td>46.82</td><td>18.84</td></tr><tr><td>SFT + traces · A/B · final RL</td><td>0.794882</td><td>32.66</td><td>39.16</td><td>42.60</td><td>44.93</td><td>46.67</td><td>48.06</td><td>49.20</td><td>50.17</td><td>51.00</td><td>51.72</td><td>52.37</td><td>52.94</td><td>53.45</td><td>53.92</td><td>54.34</td><td>54.73</td><td>0.35</td></tr><tr><td>Direct continuation · full/full · SFT B</td><td>0.727651</td><td>11.60</td><td>19.38</td><td>24.97</td><td>29.24</td><td>32.64</td><td>35.44</td><td>37.80</td><td>39.85</td><td>41.63</td><td>43.22</td><td>44.65</td><td>45.95</td><td>47.13</td><td>48.21</td><td>49.21</td><td>50.14</td><td>91.27</td></tr><tr><td>Direct continuation · full/full · final RL</td><td>0.778875</td><td>31.11</td><td>38.80</td><td>42.96</td><td>45.70</td><td>47.70</td><td>49.27</td><td>50.56</td><td>51.66</td><td>52.61</td><td>53.45</td><td>54.21</td><td>54.89</td><td>55.52</td><td>56.11</td><td>56.65</td><td>57.16</td><td>1.34</td></tr><tr><td>SFT + traces · full/full · SFT B</td><td>0.796639</td><td>26.74</td><td>34.66</td><td>38.98</td><td>41.93</td><td>44.18</td><td>45.99</td><td>47.49</td><td>48.78</td><td>49.90</td><td>50.89</td><td>51.78</td><td>52.60</td><td>53.35</td><td>54.05</td><td>54.71</td><td>55.34</td><td>17.15</td></tr><tr><td>SFT + traces · full/full · final RL</td><td>0.792758</td><td>32.97</td><td>39.47</td><td>42.92</td><td>45.24</td><td>46.98</td><td>48.35</td><td>49.47</td><td>50.42</td><td>51.23</td><td>51.95</td><td>52.59</td><td>53.16</td><td>53.68</td><td>54.16</td><td>54.59</td><td>55.00</td><td>0.47</td></tr></tbody></table></div></details>

<details><summary>Llama SFT–RL: all thirteen checkpoints, held-out loss and greedy accuracy</summary><div class="interleave-table"><table><thead><tr><th>Checkpoint</th><th>Numina SFT loss ↓</th><th>GSM8K (%)</th><th>MATH-500 (%)</th><th>Polaris raw (%)</th><th>Polaris valid-reference (%)</th></tr></thead><tbody><tr><td>Shared SFT A</td><td>0.4770</td><td>65.66</td><td>33.20</td><td>7.81</td><td>8.15</td></tr><tr><td>Baseline · complete SFT</td><td>0.4455</td><td>70.58</td><td>39.00</td><td>7.62</td><td>7.94</td></tr><tr><td>Shared first RL · A</td><td>0.6346</td><td>68.31</td><td>31.80</td><td>14.45</td><td>15.07</td></tr><tr><td>Shared first RL · full</td><td>0.6161</td><td>70.51</td><td>30.60</td><td>13.96</td><td>14.56</td></tr><tr><td>Baseline · final RL</td><td>0.7440</td><td>76.27</td><td>37.20</td><td>16.70</td><td>17.41</td></tr><tr><td>Direct continuation · A/B · SFT B</td><td>0.4441</td><td>69.52</td><td>38.80</td><td>9.86</td><td>10.29</td></tr><tr><td>Direct continuation · A/B · final RL</td><td>0.6246</td><td>74.83</td><td>40.00</td><td>16.11</td><td>16.80</td></tr><tr><td>SFT + traces · A/B · SFT B</td><td>0.4534</td><td>71.49</td><td>39.60</td><td>15.92</td><td>16.60</td></tr><tr><td>SFT + traces · A/B · final RL</td><td>0.5363</td><td>72.86</td><td>40.60</td><td>17.29</td><td>18.02</td></tr><tr><td>Direct continuation · full/full · SFT B</td><td>0.4440</td><td>70.43</td><td>39.40</td><td>9.38</td><td>9.78</td></tr><tr><td>Direct continuation · full/full · final RL</td><td>0.6218</td><td>74.07</td><td>39.60</td><td>15.62</td><td>16.29</td></tr><tr><td>SFT + traces · full/full · SFT B</td><td>0.4520</td><td>69.60</td><td>38.40</td><td>15.23</td><td>15.89</td></tr><tr><td>SFT + traces · full/full · final RL</td><td>0.5427</td><td>72.25</td><td>39.40</td><td>15.23</td><td>15.89</td></tr></tbody></table></div></details>

<h2 id="data-and-results">Data and results</h2>

The files below contain the loss, accuracy, and paired comparisons used in this post. Each SFT–RL study has five training schedules and thirteen evaluated checkpoints, counting shared checkpoints once.

- [Earlier chess checkpoint metrics](/assets/blog/interleaving-pt-rl/chess-results.json)
- [Earlier chess paired comparisons and intervals](/assets/blog/interleaving-pt-rl/chess-comparisons.json)
- [Selected OLMo2 math checkpoint metrics](/assets/blog/interleaving-pt-rl/math-results.json)
- [Chess SFT–RL: all thirteen checkpoints](/assets/blog/interleaving-pt-rl/chess-sft-results.json)
- [Llama SFT–RL: all thirteen checkpoints](/assets/blog/interleaving-pt-rl/numina-sft-results.json)
