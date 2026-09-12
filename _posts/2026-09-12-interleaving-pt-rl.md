---
layout: distill
title: "Interleaving pretraining and RL: what carries through?"
description: Experiments in chess and math track pretraining loss, intermediate gains, and what survives final reinforcement learning.
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
toc:
  - name: What we compare
  - name: First chess study
  - name: Second chess study
  - name: Math
  - name: What the experiments tell us
  - name: Evaluation and limits
  - name: Complete results
  - name: Data and results
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
---

Pretraining and reinforcement learning optimize different objectives. Pretraining rewards predicting the training data; RL rewards successful behavior on a task. A natural question is whether alternating them can make the two stages work better together.

We studied this in chess, first through a broad set of continuation and trace-training experiments, then through a second study with different rollout settings and RL data schedules. We also tested the main approaches in math.

**We have not found a consistent improvement in final task performance from interleaving.** But the intermediate checkpoints are informative: an approach can improve pretraining loss, retain more single-attempt success, or solve more problems within sixteen attempts without improving all three. Some of these gains also disappear by the end of final RL.

We organize the experiments around two questions:

1. Does inserting RL help the subsequent pretraining stage?
2. After all training is complete, is the model better than ordinary pretraining followed by RL?

<h2 id="what-we-compare">What we compare</h2>

Our conventional baseline completes pretraining and then runs 3,000 RL updates. The interleaved PT experiments divide pretraining into two halves and place 1,500 RL updates between them, followed by another 1,500 RL updates at the end. Trace-only experiments complete pretraining before the first RL stage, then insert training on successful RL trajectories between the two RL stages.

There are two ways to transfer information from RL into supervised training. We can **continue from the RL weights**, or we can **train on traces generated during RL**. These choices can be combined. A trace-training stage can also restart from the earlier PT checkpoint, in which case the traces carry the information from RL while the RL weights are discarded.

<div class="interleave-table" markdown="1">

| Study         | Model / PT budget             | RL prompts × samples per update | RL data schedule                                    |
| ------------- | ----------------------------- | ------------------------------- | --------------------------------------------------- |
| Earlier chess | 47M parameters / 5B PT tokens | 256 × 8                         | Main final comparison: 28,419 prompts               |
| Later chess   | 47M parameters / 5B PT tokens | 64 × 16                         | Full/full or disjoint A/B                           |
| Math          | OLMo2 1B / 45B PT tokens      | 64 × 16                         | Full dataset for the three approaches reported here |

</div>

Throughout the post, “PT” includes the ordinary supervised task data used by the experiment: chess SFT or math CoT data. Successful RL traces are additional data. Equal PT-token budgets and equal RL-update counts therefore do not make these comparisons equal in total compute.

We report held-out PT cross-entropy, or log loss, with lower values better. We also report sampled pass@k: the probability estimate of solving a problem in k attempts under the evaluation protocol. Pass@1 measures single-attempt success; pass@16 measures success within sixteen attempts. Neither should stand in for the other.

<h2 id="first-chess-study">First chess study: what survives the return to PT?</h2>

We began with a broad investigation of the boundary between RL and pretraining. Beyond the conventional baseline, we tested two successive PT schedules without intervening RL; training on shuffled or chronological traces; traces followed by PT; traces mixed with PT; and direct continuation from RL weights. We also tested late trace replay and different choices for retaining or resetting AdamW state.

The split-PT control matters. Dividing PT into two schedules can itself change the result. Comparing an interleaved run only with one uninterrupted PT schedule would not isolate the contribution of RL.

Consider the direct-continuation experiment with fresh AdamW. It starts from 2.5B PT tokens, runs RL, and then trains on the remaining 2.5B PT tokens without adding RL traces. Before final RL, its results are:

<div class="interleave-table" markdown="1">

| Checkpoint                   | PT loss ↓ | Pass@1 (%) | Pass@2 (%) | Pass@4 (%) | Pass@8 (%) | Pass@16 (%) |
| ---------------------------- | --------: | ---------: | ---------: | ---------: | ---------: | ----------: |
| After first RL               |  0.540289 |      30.79 |      36.39 |      41.26 |      45.60 |       49.53 |
| After direct PT2             |  0.507826 |      17.85 |      27.68 |      38.27 |      47.86 |       55.68 |
| One 5B PT schedule, no RL    |  0.512429 |      19.11 |      29.06 |      39.58 |      49.21 |       57.30 |
| Two 2.5B PT schedules, no RL |  0.507432 |      18.23 |      28.09 |      38.71 |      48.62 |       57.43 |

</div>

Returning to PT reduces pass@1 from **30.79% to 17.85%**, a loss of **12.94 percentage points**. At the same time, PT loss improves from **0.540289 to 0.507826**, and pass@16 rises from **49.53% to 55.68%**.

<figure class="l-body interleave-figure"><img src="/assets/blog/interleaving-pt-rl/direct-continuation.svg" loading="lazy" alt="Pass@k curves for the first RL checkpoint, direct PT continuation, and the 5B PT baseline."><figcaption>Figure 1. Earlier chess, direct continuation: PT lowers single-attempt success but raises success at larger k relative to the RL parent. The 5B PT baseline is shown for context.</figcaption></figure>

It would be too broad to say that PT erases all of the RL improvement. The low-k and high-k results move in different directions. The supported observation is that direct PT continuation substantially reduces the preceding RL checkpoint’s single-attempt success while improving its PT loss and success within sixteen attempts.

Does RL make the subsequent PT endpoint better? This example does not establish that. Direct continuation has slightly lower loss than the single-schedule 5B baseline, but the split-PT control without intervening RL already has slightly lower loss still: **0.507432 versus 0.507826**. Direct continuation also ends below both PT controls at the displayed pass@k values.

<h3>What changes when we include the traces?</h3>

Adding successful trajectories gives a different intermediate result. We tested both trace training from PT weights and PT-plus-trace training from RL weights.

<div class="interleave-table" markdown="1">

| Checkpoint                                | PT loss ↓ | Pass@1 (%) | Pass@2 (%) | Pass@4 (%) | Pass@8 (%) | Pass@16 (%) |
| ----------------------------------------- | --------: | ---------: | ---------: | ---------: | ---------: | ----------: |
| 5B PT baseline                            |  0.512429 |      19.11 |      29.06 |      39.58 |      49.21 |       57.30 |
| Trace-only; reset to PT weights           |  0.546131 |      37.70 |      43.32 |      47.97 |      52.03 |       55.88 |
| PT + traces; reset to PT weights          |  0.510979 |      33.50 |      40.03 |      45.52 |      50.45 |       55.00 |
| RL weights + PT + all traces; carry AdamW |  0.511704 |      34.08 |      40.40 |      45.72 |      50.67 |       55.61 |
| RL weights + PT + all traces; fresh AdamW |  0.510470 |      34.04 |      40.55 |      45.99 |      50.95 |       55.41 |

</div>

These trace-trained checkpoints have much higher observed pass@1 than the ordinary PT baseline. Their pass@16 values, however, are lower. Continuing RL weights with all successful traces reaches roughly **34% pass@1** before final RL, compared with **17.85%** for direct PT continuation without traces. This is evidence that the tested recipes produce different intermediate behavior; the optimizer and trace-source differences prevent treating every row as a one-variable ablation.

The final RL stage changes the comparison again. In the matched 28,419-prompt study, baseline ends at **37.86% pass@1 and 54.80% pass@16**. Chronological trace-only training ends at **38.49% and 53.85%**. Direct PT continuation ends at **36.36% and 54.26%**. Continuing RL weights through PT plus all traces with fresh AdamW ends at **38.01% and 54.59%**.

Across the fifteen alternatives to baseline, none of the final pass@1/pass@16 differences excludes zero after adjustment for the thirty comparisons. That is not proof of equivalence. It means this evaluation does not establish a final improvement among those alternatives. All sixteen experiments, including the optimizer and replay variants, are retained in the appendix.

<h2 id="second-chess-study">Second chess study: full data and disjoint RL stages</h2>

The later study uses 64 prompts and 16 samples per prompt. It focuses on trace-only training and PT mixed with traces, with the intermediate stages continuing from RL weights and using fresh AdamW.

We compare two data schedules. In **full/full**, both RL stages use the full set of 53,156 eligible prompts. In **A/B**, the first stage uses A and the second uses B. Each contains 26,578 prompts, and their canonical position groups are disjoint.

The A/B experiments examine whether the recipes behave differently when final RL trains on a separate subset rather than returning to the same pool. They do not by themselves isolate the effect of traces: the study does not include every possible no-trace A/B control. We also do not attribute differences between the two chess studies to rollout geometry alone, because training data and some initialization choices changed as well.

<h3>The middle-stage improvement is not the final outcome</h3>

For the full/full PT-plus-trace experiment, the first RL checkpoint has PT loss **0.536972**, pass@1 **29.17%**, and pass@16 **52.43%**. After PT plus traces, these become **0.509888**, **30.24%**, and **59.12%**. Here the middle stage improves all three observed metrics relative to its own RL parent.

Relative to the ordinary 5B PT baseline, the same checkpoint has lower PT loss, higher pass@1, and higher pass@16. That is a useful intermediate result. It does not isolate a benefit from RL itself: the earlier split-PT control has lower PT loss, and trace training adds supervised work.

Final RL increases this run’s pass@1 to **36.09%**, raises PT loss to **0.520700**, and leaves pass@16 nearly unchanged at **59.05%**. The appropriate final comparison is now the conventional PT → RL baseline:

<div class="interleave-table" markdown="1">

| Checkpoint              | PT loss ↓ | Pass@1 (%) | Pass@2 (%) | Pass@4 (%) | Pass@8 (%) | Pass@16 (%) |
| ----------------------- | --------: | ---------: | ---------: | ---------: | ---------: | ----------: |
| Baseline                |  0.534776 |      38.19 |      44.33 |      49.56 |      54.37 |       58.58 |
| Trace-only · A/B        |  0.542875 |      38.63 |      44.73 |      49.75 |      54.30 |       58.58 |
| Trace-only · full/full  |  0.543711 |      38.20 |      44.32 |      49.43 |      53.87 |       57.77 |
| PT + traces · A/B       |  0.522745 |      36.16 |      42.52 |      47.91 |      52.64 |       56.96 |
| PT + traces · full/full |  0.520700 |      36.09 |      42.56 |      48.32 |      53.74 |       59.05 |

</div>

<figure class="l-body interleave-figure"><img src="/assets/blog/interleaving-pt-rl/later-chess-final.svg" loading="lazy" alt="Three dot plots comparing final PT loss, pass@1 and pass@16 across all five later chess experiments."><figcaption>Figure 2. Later chess, after final RL. Dashed lines mark the conventional baseline. PT plus traces has lower PT loss but lower pass@1; pass@16 is mixed. These are point estimates.</figcaption></figure>

The trace-only variants finish close to baseline at pass@1, with higher PT loss. PT plus traces finishes with lower PT loss but about **two percentage points lower pass@1**. Its pass@16 outcome depends on the data schedule: **56.96%** for A/B and **59.05%** for full/full, compared with **58.58%** for baseline.

For the full/full PT-plus-trace run, the pass@1 difference is **−2.10 points**, with a multiplicity-adjusted paired-bootstrap interval of **[−3.78, −0.39]**. The pass@16 difference is **+0.47 points**, with an adjusted interval of **[−2.40, +3.24]**. The saved evaluation supports a pass@1 deficit; it does not establish a pass@16 improvement.

The two data schedules therefore do not produce a consistent final advantage from interleaving. They do reinforce the distinction between lower PT loss and better task performance.

<h2 id="math">Math: testing the main pattern in another domain</h2>

We use the completed math experiments to ask whether these tradeoffs extend beyond chess. The comparison has three approaches: a 45B PT baseline followed by RL, PT plus traces between RL stages, and direct PT continuation from RL weights.

For math PT plus traces, the middle stage restarts from PT1 weights and incorporates RL-generated traces. Direct continuation starts from RL1 weights. Both selected variants use fresh AdamW at stage boundaries. We keep these initialization choices explicit when interpreting the results.

Before final RL, the results are:

<div class="interleave-table" markdown="1">

| Approach            | PT loss ↓ | Pass@1 (%) | Pass@2 (%) | Pass@4 (%) | Pass@8 (%) | Pass@16 (%) |
| ------------------- | --------: | ---------: | ---------: | ---------: | ---------: | ----------: |
| Baseline            |  1.227608 |       9.28 |      15.08 |      22.82 |      32.05 |       42.20 |
| PT + traces         |  1.234848 |      16.83 |      24.99 |      34.09 |      43.44 |       52.60 |
| Direct continuation |  1.235861 |      10.25 |      16.50 |      24.45 |      33.52 |       43.20 |

</div>

PT plus traces has substantially higher intermediate SkyEasy success than the ordinary PT checkpoint. Direct continuation is closer to baseline. Both have slightly higher PT loss than baseline, so the observed task gains do not amount to an improvement in the pretraining objective.

After final RL:

<div class="interleave-table" markdown="1">

| Approach            | PT loss ↓ | Pass@1 (%) | Pass@2 (%) | Pass@4 (%) | Pass@8 (%) | Pass@16 (%) |
| ------------------- | --------: | ---------: | ---------: | ---------: | ---------: | ----------: |
| Baseline            |  1.240630 |      28.94 |      38.03 |      46.90 |      55.55 |       64.20 |
| PT + traces         |  1.238009 |      24.89 |      34.39 |      43.76 |      52.72 |       61.60 |
| Direct continuation |  1.242338 |      24.36 |      33.66 |      43.57 |      53.37 |       62.20 |

</div>

The conventional baseline finishes above both interleaved approaches at all five displayed SkyEasy pass@k values. The same direction appears in the additional greedy benchmarks:

<div class="interleave-table" markdown="1">

| Approach            | Numina response loss ↓ | GSM8K greedy (%) | MATH greedy (%) |
| ------------------- | ---------------------: | ---------------: | --------------: |
| Baseline            |               0.616258 |            42.53 |           31.34 |
| PT + traces         |               0.591252 |            31.31 |           28.98 |
| Direct continuation |               0.603771 |            35.63 |           28.52 |

</div>

PT plus traces has lower final PT loss and lower held-out Numina response loss than baseline, but lower final task accuracy. The direct-continuation run also finishes below baseline on task performance. These are observed endpoint differences; we do not attach training-seed uncertainty to them.

The math comparison tests the main approaches without reproducing the full chess ablation matrix. These results show that intermediate gains do not necessarily translate into stronger final RL performance in a second domain. The absence of a matched no-RL split-PT control limits what we can attribute specifically to intervening RL.

<h2 id="what-the-experiments-tell-us">What the experiments tell us</h2>

**First, returning to supervised training can change which metric looks better.** Direct PT continuation can improve loss and high-k success while reducing single-attempt success. Calling that entire transition a “washout” would hide part of the result.

**Second, traces can improve the intermediate model without delivering a better final model.** We see useful intermediate pass@k results from trace training, including when training continues from RL weights. The full PT → RL endpoint remains the comparison that answers whether the recipe improves final performance.

**Third, lower PT loss is a real measured outcome, but it is not sufficient evidence for the broader recipe.** In the later chess study and the math PT-plus-trace run, lower final PT loss coexists with lower final pass@1. Before final RL, the no-RL PT controls also challenge the claim that inserting RL itself improves pretraining.

Our conclusion is bounded by the experiments: **the interleaving recipes tested here do not provide a consistent improvement over ordinary PT followed by RL.** They reveal tradeoffs between the objectives and between single-attempt and multiple-attempt success. Establishing a mechanism for those tradeoffs would require evidence beyond these endpoint metrics.

<h2 id="evaluation-and-limits">Evaluation and limits</h2>

Chess evaluates 1,480 held-out puzzles with sixteen samples each, temperature 1 and top-p 1, using a 2,048-token context with a 512-token prompt cap and a 1,536-model-token response budget. Four overlength puzzles from the original 1,484 are excluded by the frozen admission rule. PT loss is token cross-entropy on 8,388,608 held-out targets.

For c successful samples out of n = 16, we calculate pass@k per problem as <code>1 − C(n−c,k) / C(n,k)</code>, then average across problems. Pass@16 is success within the saved sixteen attempts; it is not a measure of unlimited capability or a direct diversity metric.

Math SkyEasy evaluation uses 500 held-out questions and sixteen samples per question at temperature 1 and top-p 1, with native context 4,096. GSM8K and MATH results use separate greedy evaluations on 1,319 and 5,000 questions. Numina response loss uses 100 held-out questions with prompt tokens masked. PT loss aggregates the saved held-out PT components; its numerical scale is not comparable with chess loss.

The chess final comparisons use 20,000 paired bootstrap replicates over puzzle identities. The reported adjusted intervals account for eight contrasts in the later study and thirty in the earlier study. These intervals are conditional on the trained checkpoints and saved generations. They do not include independent training-seed variation. We do not report corresponding intervals for the intermediate comparisons or math results here, or population confidence intervals for PT loss.

All results here come from saved evaluations. The chess analysis uses the corrected evaluation lineage and preserves data-matched final endpoints. Experiments differ in trace data, learning-rate schedules, optimizer handling, and sometimes RL data; these are comparisons of specified training recipes, not a complete causal decomposition. Total FLOPs are not matched. Missing controls limit the claims but do not prevent reporting the completed results.

<h2 id="complete-results">Complete results</h2>

The tables below retain every saved chess stage in the two studies, including the A/B variants and additional final-RL dataset variants. Main-text comparisons use final checkpoints matched to their study’s baseline. Values are rounded for display; the source files retain full precision.

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
<details><summary>Earlier chess: six additional final endpoints on the larger RL set</summary><p>These endpoints use 53,225 RL training prompts and are not substituted for the 28,419-prompt main comparison.</p><div class="interleave-table"><table>
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
<details><summary>Every saved 256x8 chess checkpoint · full pass@1–16</summary><div class="interleave-table"><table>
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
<details><summary>Every saved 64x16 chess checkpoint · full pass@1–16</summary><div class="interleave-table"><table>
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

<h2 id="data-and-results">Data and results</h2>

The complete chess tables are included above. These downloads preserve the numerical results used in this post, including full pass@1–16 values and the final chess comparison intervals. Results reflect saved evaluations collected through September 10, 2026.

- [Chess checkpoint metrics](/assets/blog/interleaving-pt-rl/chess-results.json)
- [Chess final paired comparisons](/assets/blog/interleaving-pt-rl/chess-comparisons.json)
- [Selected math checkpoint metrics](/assets/blog/interleaving-pt-rl/math-results.json)
