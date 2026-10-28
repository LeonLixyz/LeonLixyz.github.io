---
layout: distill
published: true
title: "An Exploration of Interleaving Pretraining and Reinforcement Learning"
title_html: "An Exploration of Interleaving<br>Pretraining and Reinforcement Learning"
description: "We test whether an intermediate RL stage during pretraining improves learning from subsequent data, or improves the final model under a fixed training budget. In chess and math, it improves neither."
date: 2026-10-27
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
bibliography: interleaving-pt-rl.bib
_styles: |
  html body d-title h1, html body d-title p, html body d-byline .byline { grid-column: middle-start / middle-end; }
  html body d-article > *, html body d-article > .l-body, html body d-article > .l-page, html body d-appendix > * { grid-column: middle-start / middle-end; }
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
  d-article .interleave-stage-strip { margin: 0.4rem 0 1.4rem; }
  d-article .interleave-stage-strip svg { display: block; width: 100%; max-width: 566px; height: auto; }
  d-article details { margin-bottom: 1rem; max-width: 100%; overflow-x: auto; }
  d-article details summary { cursor: pointer; font-weight: 600; }
  d-article details .interleave-table { margin-bottom: 0; }
  d-article h2 { scroll-margin-top: 80px; }
  d-article .interleave-text-table td, d-article .interleave-text-table th { white-space: normal !important; text-align: left !important; }
  d-article mjx-container[display="true"] { overflow-x: auto; overflow-y: hidden; max-width: 100%; padding: 0.7rem 0; }
  d-article .interleave-summary { border-left: 3px solid #b66b35; padding: 0.2rem 0 0.2rem 1.1rem; margin: 1.5rem 0; }
  d-article .interleave-summary ol { margin-bottom: 0; }
  d-byline a:not(.navbar-brand):not(.nav-link):not(.publication-link) { --link-color: var(--global-text-color); --link-hover-bg: transparent; color: var(--global-text-color) !important; }
  d-byline a:not(.navbar-brand):not(.nav-link):not(.publication-link):hover { color: var(--global-text-color) !important; text-decoration: underline !important; box-shadow: none; transform: none; }
  html body d-article a[href], html body d-appendix a[href], html body d-title a[href] { --link-color: var(--global-theme-color); --link-hover-bg: rgba(0, 119, 200, 0.1); }
  d-article .interleave-setup-table th:last-child, d-article .interleave-setup-table td:last-child { white-space: nowrap !important; }
  d-article .interleave-note { font-size: 0.85rem; color: var(--global-text-color-light, #6c757d); margin-top: -0.6rem; }
---

Language models are usually trained in a fixed sequence of stages. Pretraining comes first and minimizes the next-token prediction loss on a large, fixed dataset, and reinforcement learning (RL), most often GRPO, comes last. In RL, the model samples several responses per prompt, each response receives a reward, and the policy is updated to increase the likelihood of responses with positive advantage.

Under this ordering, RL begins only after all pretraining and supervised finetuning are complete. Human learning is rarely organized this way. A student does not attend every lecture of a course before attempting the first problem set; lectures and exercises alternate, and working through problems often makes the next lecture easier to follow. We ask whether language models benefit from the same kind of alternation, that is, whether interleaving RL with pretraining helps more than confining RL to the end.

In principle, RL does not have to wait until pretraining ends, since RL can improve performance on a task as soon as the model's pass@k on that task is nonzero. Moreover, different skills become learnable at different times. The [quantization model of neural scaling](https://arxiv.org/abs/2303.13506) describes a model's abilities as discrete skills that are learned at different points in training, roughly in order of how often the data require them. In a math dataset with problems at the level of GSM8K, AIME, and the IMO, for example, we would expect the model to solve GSM8K-level problems first, AIME-level problems later, and IMO-level problems last, so each task becomes trainable with RL at a different point during pretraining. If RL improves a distribution of tasks before pretraining ends, the improved skills might make the remaining pretraining data easier to learn.

Empirically, RL is already effective on intermediate pretraining checkpoints. In our previous paper, [Understanding Reasoning from Pretraining to Post-Training](https://arxiv.org/abs/2607.16097), we ran RL from many pretraining checkpoints of chess models with 20M to 680M parameters ([Figure 1](#figure-1)). Once a checkpoint had been pretrained on enough tokens, RL improved its accuracy, and the more pretraining a checkpoint had received, the higher its accuracy after RL. [RL Excursions during Pre-Training](https://arxiv.org/abs/2606.04272) reports the same for early pretraining checkpoints of a 1B model pretrained from scratch.

<figure id="figure-1" class="l-page interleave-figure"><a href="/assets/blog/interleaving-pt-rl/prior-work-early-rl.svg" target="_blank" aria-label="Open the figure at full size"><img src="/assets/blog/interleaving-pt-rl/prior-work-early-rl.svg" loading="lazy" alt="Eight panels for chess models of 20M, 50M, 200M and 680M parameters, showing pass at 1 and pass at 16 against total compute. Each RL run starts from a pretraining checkpoint colored by its pretraining compute. RL raises pass at 1 from all but the earliest checkpoints, and runs started from early checkpoints form the frontier at small compute budgets."></a><figcaption>Figure 1. RL from pretraining checkpoints of chess models with 20M, 50M, 200M, and 680M parameters, from our previous paper. Each colored curve is one RL run; its color gives the pretraining compute of the checkpoint it starts from, and open circles mark that checkpoint before RL. The dashed black curve is the best pass@1 or pass@16 reached at each total compute, and the percentages give the share of compute spent on RL where a run first reaches it. Reproduced from <a href="https://arxiv.org/abs/2607.16097">Shen et al.</a></figcaption></figure>

Several recent training pipelines already alternate between the two objectives. MAI-Thinking-1, for example, fine-tunes a midtrained checkpoint on the order of 1M successful responses collected during RL and resumes RL from the result, a step its report calls self-distillation. This resets numerical instabilities that accumulate in the parameters before an RL run diverges, and it carries RL progress over to newer pretrained and midtrained checkpoints ([report, §3.1.4](https://microsoft.ai/pdf/mai-thinking-1.pdf#page=34)). Ring-Zero selects and shortens correct RL outputs, fine-tunes the original base model on them, and continues RL ([paper, §3.1.2](https://arxiv.org/html/2607.12395v2#S3.SS1.SSS2)). ReMiT uses an RL-tuned model to reweight token losses during midtraining ([paper, §3](https://arxiv.org/html/2602.03075v1#S3)). However, to our knowledge, no prior work directly tests whether interleaving pretraining with RL helps. In this post, we directly study this.

In our experiments, we split both the pretraining data and the RL updates into two halves, which gives four stages, PT 1, RL 1, PT 2, and RL 2. The **standard** baseline pretrains on all the data and then performs all the RL updates. Our interleaved schedule, **Inter-RL**, runs the four stages in consecutive order.

<div class="interleave-stage-strip"><svg viewBox="0 0 566 104" role="img" aria-label="Standard baseline: all pretraining, then one RL stage with all updates. Inter-RL: PT 1, then RL 1, then PT 2, then RL 2."><style>.b{font:600 14px Roboto,"Helvetica Neue",Arial,sans-serif;fill:#fff}.l{font:500 14px Roboto,"Helvetica Neue",Arial,sans-serif;fill:var(--global-text-color,#333)}.a{stroke:#8a97a1;stroke-width:1.8}.h{fill:#8a97a1}</style><text x="0" y="30" class="l">Standard</text><rect x="120" y="8" width="248" height="34" rx="4" fill="#6f7f8c"/><text x="244" y="30" text-anchor="middle" class="b">PT</text><rect x="396" y="8" width="168" height="34" rx="4" fill="#2f7fb8"/><text x="480" y="30" text-anchor="middle" class="b">RL</text><line x1="372" y1="25" x2="387" y2="25" class="a"/><path d="M386,20 L393,25 L386,30 Z" class="h"/><text x="0" y="84" class="l">Inter-RL</text><rect x="120" y="62" width="110" height="34" rx="4" fill="#6f7f8c"/><text x="175" y="84" text-anchor="middle" class="b">PT 1</text><rect x="258" y="62" width="70" height="34" rx="4" fill="#2f7fb8"/><text x="293" y="84" text-anchor="middle" class="b">RL 1</text><line x1="234" y1="79" x2="249" y2="79" class="a"/><path d="M248,74 L255,79 L248,84 Z" class="h"/><rect x="356" y="62" width="110" height="34" rx="4" fill="#6f7f8c"/><text x="411" y="84" text-anchor="middle" class="b">PT 2</text><line x1="332" y1="79" x2="347" y2="79" class="a"/><path d="M346,74 L353,79 L346,84 Z" class="h"/><rect x="494" y="62" width="70" height="34" rx="4" fill="#2f7fb8"/><text x="529" y="84" text-anchor="middle" class="b">RL 2</text><line x1="470" y1="79" x2="485" y2="79" class="a"/><path d="M484,74 L491,79 L484,84 Z" class="h"/></svg></div>

We study two questions:

1. **Does an intermediate RL stage improve learning from subsequent data?** If RL 1 makes the PT 2 data easier to learn, the interleaved model should reach a lower held-out pretraining loss at the end of PT 2 than a model trained on the same data without RL.
2. **Does an intermediate RL stage improve the final model under a fixed training budget?** We compare the final models after both have used the same pretraining data and the same total number of RL updates.

We study these questions in chess and math, with models ranging from a 47M-parameter chess transformer to OLMo2 1B. The answer to both questions is no.

<h2 id="tl-dr">TL;DR</h2>

<div class="interleave-summary" markdown="1">

1. **An intermediate RL stage does not improve pretraining.** At the end of pretraining, the held-out pretraining loss is the same as without RL in chess (0.5078 vs. 0.5074) and slightly higher in math (1.2359 vs. 1.2276), as the loss panels of [Figure 3](#figure-3) show.
2. **Continued pretraining after RL erases the RL gain.** In chess, RL 1 raises pass@1 from 13.9% to 30.8%. By the end of PT 2, which trains on the second 2.5B tokens of pretraining data, pass@1 is back to 17.9%, close to the 18.2% of a model trained on the same data without RL ([Figure 2](#figure-2)).
3. **Replaying RL's correct responses keeps the RL gain but does not improve pretraining.** When we add the correct responses sampled during RL 1 to the PT 2 data (PT 2 + replay), the model keeps the RL gain, and its held-out pretraining loss is unchanged or slightly higher ([Figure 6](#figure-6)).
4. **Interleaving does not improve the final model.** At the end of PT 2, the replay runs are well ahead of the baseline, but only because they have already spent half of their RL budget, whereas the baseline has not started RL. After all 3,000 RL updates, the replay runs end level with the baseline in chess (38.0% vs. 37.9%) and below it in math (24.9% vs. 28.9%), as the RL 2 bars of [Figure 2](#figure-2) show.
5. **Interleaving SFT and RL gives mixed results.** SFT data are much closer to RL responses than pretraining text is, yet the RL gain is again forgotten without replay, and replay again leaves the held-out loss unchanged or higher. In chess SFT, every interleaved schedule ends below the baseline, while in Llama the interleaved schedules gain on MATH-500, lose on GSM8K, and stay roughly level on Polaris, the RL task ([Figures 9](#figure-9) and [10](#figure-10)).

</div>

<figure id="figure-2" class="l-page interleave-figure"><a href="/assets/blog/interleaving-pt-rl/head-start-summary.svg" target="_blank" aria-label="Open the summary figure at full size"><img src="/assets/blog/interleaving-pt-rl/head-start-summary.svg" loading="lazy" alt="Four bar charts, one per experiment, showing task accuracy at the end of each training stage: PT 1, RL 1, PT 2, and RL 2, with SFT 1 and SFT 2 in the SFT experiments. Inter-RL traces a zigzag: accuracy rises in RL 1, falls back close to the baseline in PT 2, and rises again in RL 2. With replay the gain is kept through PT 2, and at the end of RL 2 the standard baseline matches or exceeds the interleaved schedules in every panel except Llama, where replay finishes 0.6 points ahead."></a><figcaption>Figure 2. Task accuracy at the end of each training stage. Inter-RL rises in RL 1, falls back close to the baseline in PT 2, and rises again in RL 2. RL 1 and RL 2 have 1,500 updates each, and the standard baseline has no RL 1, so its single RL stage of 3,000 updates appears in the RL 2 group. The first bar in each panel is the PT 1 or SFT 1 checkpoint that all schedules share. In math PT and math SFT, the baseline's PT 2 bar comes from a single run over all the data. The Llama replay schedule uses split prompts, and math PT's replay run starts PT 2 from the PT 1 checkpoint. The Llama panel shows greedy accuracy on held-out Polaris, its RL task, and the other panels show pass@1.</figcaption></figure>

<h2 id="setup">Setup</h2>

We run two pretraining experiments, chess PT and math PT. In both, we mix SFT data, that is, worked solutions to the task, uniformly into the pretraining data and split the mixture into two halves, PT 1 and PT 2.

<div class="interleave-table interleave-text-table interleave-setup-table" markdown="1">

| Experiment | Model | Pretraining data | RL task · prompts × group size |
| --- | --- | --- | --- |
| Chess PT | 47M | 5B pretraining tokens + 77,717 worked puzzle solutions, each included 3 times | Chess puzzles · 256 × 8 |
| Math PT | OLMo2 1B | 45B pretraining tokens + the NuminaMath-CoT solutions, each included once | SkyEasy math · 64 × 16 |

</div>

RL 1 and RL 2 have 1,500 updates each, and the standard baseline performs all 3,000 in one stage. By default, both RL stages sample from the full prompt set, since several epochs over the same prompts are standard in RL training. As a check, we also run some schedules with **split prompts**, in which the prompt set is divided into two disjoint halves, one per RL stage, so that RL 2 sees only prompts that RL 1 did not.

Every schedule uses the same pretraining data and the same 3,000 RL updates. However, the budgets match only at the end of each experiment. At the end of PT 2, inter-RL has already used 1,500 RL updates and the standard baseline none, so inter-RL is ahead at that point by construction. Moreover, replay, which we introduce below to counter catastrophic forgetting, trains on the replayed RL responses and therefore uses more compute.

<h2 id="does-an-intermediate-rl-stage-improve-learning-from-subsequent-data">Does an intermediate RL stage improve learning from subsequent data?</h2>

<h3>Inter-RL: the RL gain is forgotten</h3>

We start with chess PT, in which PT 1 and PT 2 each contain 2.5B tokens and RL 1 consists of 1,500 updates. Inter-RL continues PT 2 from the RL 1 checkpoint with a newly initialized optimizer, while the standard baseline runs PT 1 and PT 2 as two separate runs, so that the two schedules differ only by RL 1 (see the note on the learning-rate schedule below). [Figure 3](#figure-3) plots the held-out pretraining loss, pass@1, and pass@16 at the end of each stage.

<figure id="figure-3" class="l-page interleave-figure"><a href="/assets/blog/interleaving-pt-rl/stage-trajectory.svg" target="_blank" aria-label="Open the stage figure at full size"><img src="/assets/blog/interleaving-pt-rl/stage-trajectory.svg" loading="lazy" alt="Two rows of three panels, chess PT and math PT, showing pretraining loss, pass at 1, and pass at 16 at the end of each training stage for the standard baseline, inter-RL, and inter-RL with replay. RL 1 raises pass at 1 sharply, PT 2 returns it to the baseline level, and the pretraining loss at the end of PT 2 is equal to or higher than the baseline."></a><figcaption>Figure 3. Held-out pretraining loss, pass@1, and pass@16 at the end of PT 1, RL 1, and PT 2. The standard baseline has no RL 1. RL 2 is shown in [Figure 7](#figure-7). In chess PT, the baseline runs PT 1 and PT 2 as two runs; in math PT, it is a single 45B-token run.</figcaption></figure>

In chess PT, RL 1 raises pass@1 from 13.85% at the end of PT 1 to 30.79% ([Figure 3](#figure-3)). However, by the end of PT 2, pass@1 drops to **17.85%**, slightly below the 18.23% of the standard baseline, and the held-out pretraining loss converges to nearly the same value, 0.507826 with inter-RL and 0.507432 without. By both measures, the model at the end of PT 2 is indistinguishable from one that never ran RL. In other words, PT 2 erases the RL gain, an instance of catastrophic forgetting, and RL 1 does not make the pretraining data easier to learn. Across the stages, inter-RL therefore traces a zigzag in which pass@1 rises in RL 1, falls back close to the baseline in PT 2, and rises again in RL 2, and the same zigzag appears in every experiment, including both SFT experiments ([Figure 2](#figure-2)).

Math PT shows the same pattern in the bottom row of [Figure 3](#figure-3). RL 1 raises SkyEasy pass@1 from 5.36% to 16.76%, but by the end of PT 2, which trains on the second 22.5B tokens, it is down to 10.25%, against 9.28% for the baseline, so about one point of the RL gain remains. The held-out pretraining loss is higher than the baseline's, 1.2359 vs. 1.2276.

**Learning-rate schedule and optimizer state.** A confounding factor is that inter-RL resets its optimizer state between PT 1 and PT 2 and has two learning-rate decays instead of one ([Figure 4](#figure-4)). To test this, in chess PT we also pretrain the standard baseline as two separate runs, PT 1 and PT 2, each with a new optimizer and its own learning-rate decay; this two-run version is the baseline used above. The two-run and one-run baselines finish almost identically, at 37.94% and 37.86% final pass@1. Moreover, we test whether PT and RL should share the same optimizer state instead of resetting it at each stage, and again find no difference.

<figure id="figure-4" class="l-body interleave-figure"><picture><source media="(max-width: 600px)" srcset="/assets/blog/interleaving-pt-rl/schedules-reset-compact.svg"><img src="/assets/blog/interleaving-pt-rl/schedules-reset.svg" loading="lazy" alt="Three schedules. Standard with one pretraining run: PT, then 3,000 RL updates. Standard with two pretraining runs: PT 1, PT 2, then 3,000 RL updates, Inter-RL: PT 1, RL 1, PT 2, RL 2. Every stage starts a new optimizer and learning-rate schedule."></picture><figcaption>Figure 4. The two standard baselines and inter-RL. Dark bars mark the start of a new optimizer and learning-rate schedule. The two-run baseline has the same reset between PT 1 and PT 2 as inter-RL, so the two differ only by RL 1. Block widths indicate order and data size, not compute.</figcaption></figure>

**Pass@16.** RL 1 improves single-attempt accuracy but not coverage, as the pass@1 and pass@16 panels of [Figure 3](#figure-3) show. In chess PT, it raises pass@1 from 13.85% to 30.79% but slightly lowers pass@16, from 51.01% to 49.53%. By the end of PT 2, inter-RL is close to the baseline on both metrics, with 17.85% vs. 18.23% pass@1 and 55.68% vs. 57.43% pass@16.

<h3>Inter-RL + replay: the RL gain is retained, but pretraining does not improve</h3>

A standard remedy for catastrophic forgetting is replay, that is, retaining examples from the earlier task in later training. In **Inter-RL + replay**, we shuffle the correct responses sampled during RL 1 uniformly into the PT 2 data, which makes PT 2 correspondingly longer ([Figure 5](#figure-5)). We call this stage PT 2 + replay.

<figure id="figure-5" class="l-body interleave-figure"><picture><source media="(max-width: 600px)" srcset="/assets/blog/interleaving-pt-rl/schedules-replay-compact.svg"><img src="/assets/blog/interleaving-pt-rl/schedules-replay.svg" loading="lazy" alt="Inter-RL: PT 1, RL 1, PT 2, RL 2. Inter-RL plus replay: the same, with the correct responses from RL 1 shuffled uniformly into PT 2, which makes that stage longer."></picture><figcaption>Figure 5. Inter-RL with and without replay. Both use the same pretraining data and 3,000 RL updates in total; the replayed responses make PT 2 longer. Block widths indicate order and data size, not compute.</figcaption></figure>

With replay, the model keeps the RL gain. In both experiments, the replay run ends PT 2 above the RL 1 checkpoint whose responses it replayed, at 34.04% vs. 30.79% pass@1 in chess PT and 16.83% vs. 16.76% in math PT ([Figure 6](#figure-6)). However, the held-out pretraining loss does not improve, reaching 0.5105 in chess PT against 0.5074 for the baseline, and 1.2348 in math PT against 1.2276 (loss panels of [Figure 6](#figure-6)).

<figure id="figure-6" class="l-page interleave-figure"><a href="/assets/blog/interleaving-pt-rl/stage-trajectory-replay.svg" target="_blank" aria-label="Open the figure at full size"><img src="/assets/blog/interleaving-pt-rl/stage-trajectory-replay.svg" loading="lazy" alt="The stage-by-stage figure for chess PT and math PT with the inter-RL plus replay schedule added. Replay keeps pass@1 high through PT 2, but its pretraining loss at the end of PT 2 is no lower than the baseline's."></a><figcaption>Figure 6. Figure 3 with Inter-RL + replay added. In chess PT, the replay run continues PT 2 from the RL 1 checkpoint; in math PT, it starts PT 2 from the PT 1 checkpoint, so RL 1 contributes only the replayed responses.</figcaption></figure>

PT 2 + replay can also start from the PT 1 checkpoint instead of the RL 1 checkpoint, so that RL 1 contributes only the replayed responses and none of its weight updates. In chess PT, this variant keeps most of the RL gain but ends slightly lower, at 33.50% pass@1 at the end of PT 2 (vs. 34.04%) and 37.14% at the end of RL 2 (vs. 38.01%). Math PT uses only this variant.

**Answer to the first question: no.** When PT 2 continues from the RL 1 checkpoint, the RL gain is forgotten and the held-out pretraining loss converges to its value without RL. With replay, the RL gain is retained, but the held-out pretraining loss is no lower.

<h2 id="does-an-intermediate-rl-stage-improve-the-final-model">Does an intermediate RL stage improve the final model?</h2>

At the end of PT 2, the replay runs are well ahead of the baseline ([Figure 6](#figure-6)). The question is whether this lead survives once every schedule has used all 3,000 RL updates.

In training reward, the lead persists through RL 2 in chess PT but not in math PT ([Figure 7](#figure-7)). In chess PT, the replay run starts RL 2 at about the reward the baseline reaches after 1,500 updates and stays slightly above it. In math PT, it starts RL 2 well below the baseline and does not close the gap. Inter-RL starts RL 2 far below the reward it reached at the end of RL 1, because PT 2 has erased the RL gain, and finishes below the baseline in both experiments.

<figure id="figure-7" class="l-page interleave-figure"><a href="/assets/blog/interleaving-pt-rl/rl-reward-curves.svg" target="_blank" aria-label="Open the figure at full size"><img src="/assets/blog/interleaving-pt-rl/rl-reward-curves.svg" loading="lazy" alt="Training reward against the number of RL updates used so far, for chess PT and math PT. The standard baseline rises over 3,000 updates. Inter-RL rises in RL 1, restarts much lower after PT 2, and finishes below the baseline. Inter-RL plus replay starts RL 2 near the baseline in chess and below it in math."></a><figcaption>Figure 7. Training reward, the fraction of sampled responses that are correct on the RL prompts, against the number of RL updates a schedule has used so far (50-update moving average). The interleaved schedules run PT 2 after update 1,500, so their RL 2 curves start there. The two interleaved schedules share RL 1.</figcaption></figure>

On held-out problems, the lead disappears in both experiments. Over its 3,000 RL updates, the standard baseline gains 19–20 points of pass@1, from 18.2% to 37.9% in chess and from 9.3% to 28.9% in math. The replay runs start RL 2 higher, at 34.0% in chess and 16.8% in math ([Figure 6](#figure-6)), but over their 1,500 updates they gain only 4.0 and 8.1 points. As a result, they finish level with the baseline in chess PT (38.0% vs. 37.9%) and 4.1 points below it in math PT, and inter-RL without replay finishes below the baseline in both experiments, at 36.4% vs. 37.9% in chess PT and 24.4% vs. 28.9% in math PT ([Figure 8](#figure-8)). An intermediate RL stage therefore changes when the model reaches its final accuracy, but not the accuracy it reaches.

<figure id="figure-8" class="l-page interleave-figure"><a href="/assets/blog/interleaving-pt-rl/pt-final-models.svg" target="_blank" aria-label="Open the figure at full size"><img src="/assets/blog/interleaving-pt-rl/pt-final-models.svg" loading="lazy" alt="Final accuracy after all 3,000 RL updates. Chess PT pass@1: standard 37.9, inter-RL 36.4, inter-RL plus replay 38.0. Math PT: the standard baseline is highest on SkyEasy, GSM8K and MATH."></a><figcaption>Figure 8. Final models of the pretraining experiments, after all 3,000 RL updates. Chess PT and SkyEasy report pass@1 estimated from 16 samples per problem; GSM8K and MATH report greedy accuracy. Single runs per schedule in math PT.</figcaption></figure>

In chess PT, we compute a 95% confidence interval for each difference from the baseline with a paired bootstrap over puzzles (20,000 resamples, adjusted for multiple comparisons). Every interval contains zero, so these chess differences could be noise. Math PT has one run per schedule, and every interleaved schedule finishes below the baseline on every benchmark ([Figure 8](#figure-8)), with SkyEasy pass@1 at 24.4–24.9% vs. 28.9%, GSM8K at 31.3–35.6% vs. 42.5%, and MATH at 28.5–29.0% vs. 31.3%.

**Answer to the second question: no.** Under the same training budget, the standard baseline ends level with or ahead of every interleaved schedule in both pretraining experiments. We return to SFT below, where the results are mixed.

We also ran the chess PT replay schedules with a different RL batch shape, 64 prompts × 16 responses per update instead of 256 × 8. Replay again keeps the RL gain through PT 2 (30.2% vs. 19.1% pass@1), and at the end of RL 2 the baseline is again ahead (38.2% vs. 36.1%); here the paired bootstrap intervals exclude zero.

<h2 id="interleaving-sft-and-rl">Interleaving SFT and RL</h2>

Pretraining data and RL responses are very different, in both the distribution of tasks and the format of the text, so next-token training on pretraining data may pull the policy away from the responses that RL reinforced. SFT data serve as the warmup for RL and are therefore much closer to RL responses, as both are reasoning traces for the same tasks. If this difference were the cause of forgetting, interleaving SFT and RL should preserve the RL gain and make the SFT data easier to learn. We test this by running the same three schedules with SFT in place of pretraining, splitting the SFT data into two halves, SFT 1 and SFT 2, and starting all runs in an experiment from the same pretrained model.

<div class="interleave-table interleave-text-table interleave-setup-table" markdown="1">

| Experiment | Model | SFT data | RL task · prompts × group size |
| --- | --- | --- | --- |
| Chess SFT | 47M, pretrained on 9.18B tokens | 77,717 worked puzzle solutions | Chess puzzles · 64 × 16 |
| Math SFT | Llama 3.2 3B | 823,505 NuminaMath-CoT solutions | Polaris math · 64 × 16 |

</div>

<h3>Chess SFT</h3>

The closer distribution does not prevent forgetting. RL 1 raises pass@1 from 2.8% to 25.5%, but at the end of SFT 2, inter-RL is at 11.6%, against 11.4% for the baseline, so SFT 2 erases the entire RL gain ([Figure 9](#figure-9)). Although SFT data share the format of RL responses, they are still off-policy, since they are sampled from other models rather than from the current model, so next-token training on them moves the policy away from the responses that RL reinforced, as pretraining does. With replay, pass@1 stays at 26.7%; however, the held-out pretraining loss rises to 0.7966, against 0.7222 for the baseline. After RL 2, the baseline has the highest pass@1, 33.5%, while inter-RL finishes 2.4 points below it and replay 0.6 points below it ([Figures 2](#figure-2) and [9](#figure-9)). The training reward ends lower as well, at 41.6% for inter-RL and 44.1% for replay against 45.2% for the baseline, averaged over the last 50 updates.

<figure id="figure-9" class="l-page interleave-figure"><a href="/assets/blog/interleaving-pt-rl/sft-chess.svg" target="_blank" aria-label="Open the figure at full size"><img src="/assets/blog/interleaving-pt-rl/sft-chess.svg" loading="lazy" alt="Chess SFT at the end of SFT 1, RL 1, SFT 2 and RL 2 for the standard baseline, inter-RL and inter-RL plus replay: pretraining loss, pass@1 and pass@16, plus the RL training reward against the number of RL updates used so far. Inter-RL rises in RL 1 and falls back to the baseline in SFT 2; replay keeps the RL gain."></a><figcaption>Figure 9. Chess SFT at the end of each stage, and the RL training reward against the number of RL updates a schedule has used so far (50-update moving average). The standard baseline has no RL 1, so its stage line runs directly from SFT 1 to SFT 2. Both interleaved schedules use the same prompts in RL 1 and RL 2 and share RL 1. The pretraining loss is measured on held-out pretraining text.</figcaption></figure>

<h3>Math SFT (Llama 3.2 3B)</h3>

RL 1 raises Polaris accuracy from 7.8% to 14.0%, and SFT 2 erases most of the gain, leaving inter-RL at 9.4% against 7.6% for the baseline ([Figure 10](#figure-10)). With replay, Polaris accuracy stays at 15.9%; however, the held-out NuminaMath loss is higher than the baseline's, 0.4534 vs. 0.4455. Without replay, the loss is 0.0015 lower than the baseline's (0.4440 vs. 0.4455), but these runs reinitialize the optimizer between SFT 1 and SFT 2 while the baseline does not, so we do not read this as an improvement.

The final models differ only slightly. On Polaris, inter-RL finishes 1.1 points below the baseline and replay 0.6 points above it. On GSM8K and MATH-500, all three schedules are within 2 points of each other at the end of SFT 2, and the final RL stage separates them. It raises GSM8K by 4.6 points in the baseline but by only 1.2 points in the replay run, likely because its SFT 2 already trained on the correct responses of RL 1. It also lowers MATH-500 in the baseline, from 39.2% to 37.2%, whereas RL 2 does not lower it in either interleaved schedule; we did not evaluate intermediate RL checkpoints, so we cannot say why. As a result, inter-RL and replay finish 2.6 and 3.4 points above the baseline on MATH-500 and 1.1 and 2.6 points below it on GSM8K. To check whether these differences exceed the sampling noise of a finite benchmark, we compute a 95% confidence interval for each difference by comparing the two models question by question. Only the replay run's GSM8K deficit has an interval that does not contain zero; every other interval contains zero, so those differences could be noise. Each schedule is also a single training run.

<figure id="figure-10" class="l-page interleave-figure"><a href="/assets/blog/interleaving-pt-rl/sft-math.svg" target="_blank" aria-label="Open the figure at full size"><img src="/assets/blog/interleaving-pt-rl/sft-math.svg" loading="lazy" alt="Math SFT on Llama 3.2 3B at the end of SFT 1, RL 1, SFT 2 and RL 2 for the standard baseline, inter-RL and inter-RL plus replay: NuminaMath loss and accuracy on Polaris, GSM8K and MATH-500, plus the RL training reward against the number of RL updates used so far."></a><figcaption>Figure 10. Math SFT on Llama 3.2 3B at the end of each stage, and the RL training reward against the number of RL updates a schedule has used so far (50-update moving average). The standard baseline has no RL 1. Inter-RL uses the same prompts in both RL stages; the replay run uses split prompts and therefore has its own RL 1. Accuracy is greedy pass@1. Single runs.</figcaption></figure>

The interleaved Llama runs end with a much lower NuminaMath loss than the baseline, for example 0.5363 vs. 0.7440 ([Figure 10](#figure-10)). This does not reflect more effective SFT, since at the end of SFT 2 the same replay run has a slightly higher loss than the baseline. RL raises the loss on the SFT data, and the baseline has received 3,000 consecutive RL updates since its last SFT step, whereas the interleaved runs have received only the 1,500 of RL 2.

<h2 id="conclusions">Conclusions</h2>

Across chess and math, and with either pretraining or SFT between the RL stages, we find no benefit from interleaving RL with the rest of training. An intermediate RL stage does not make the subsequent data easier to learn, and under the same data and RL budget the standard schedule ends level with or ahead of every interleaved schedule.

These results are consistent with our [previous paper](https://arxiv.org/abs/2607.16097), which found that pretraining predicts how RL scales ([Figure 11](#figure-11)). Across chess models with 20M to 680M parameters, a lower pretraining loss gives a higher reward at every level of RL compute, and more pretraining tokens give a larger reward gain per tenfold increase in RL compute. The reward curves of the pretraining experiments in this post show the same dependence ([Figure 7](#figure-7)). RL 1 starts from the checkpoint halfway through pretraining and rises more slowly than the baseline's RL, which starts after all pretraining. After 1,500 updates, the training reward of RL 1 is 70.8% against 78.7% for the baseline in chess, and 17.9% against 29.4% in math, averaged over the last 50 updates. RL 2 of inter-RL also starts after all pretraining, and its reward follows the baseline's first 1,500 updates almost exactly, ending at 78.9% vs. 78.7% in chess and 29.4% vs. 29.4% in math. RL 2 therefore gains nothing from the 1,500 updates of RL 1, and its reward curve matches that of an RL run started after the same amount of pretraining.

<figure id="figure-11" class="l-page interleave-figure"><a href="/assets/blog/interleaving-pt-rl/prior-work-pretraining-predicts-rl.svg" target="_blank" aria-label="Open the figure at full size"><img src="/assets/blog/interleaving-pt-rl/prior-work-pretraining-predicts-rl.svg" loading="lazy" alt="Three panels from the previous paper. Left: fitted chess puzzle pass@1 at five reference levels of RL compute against pretraining eval loss; lower loss gives higher reward at every level. Middle: the slope of reward per tenfold increase in RL compute against log pretraining tokens; more tokens give a larger slope. Right: observed against predicted slope from a joint fit on pretraining tokens and model size."></a><figcaption>Figure 11. Pretraining predicts RL scaling in chess models with 20M to 680M parameters, reproduced from <a href="https://arxiv.org/abs/2607.16097">Shen et al.</a> Left, fitted pass@1 after RL at five reference levels of RL compute, against pretraining eval loss. Middle, slope of pass@1 per tenfold increase in RL compute, against pretraining tokens. Right, observed slope against the slope predicted from pretraining tokens and model size. Each marker is one pretraining checkpoint followed by RL.</figcaption></figure>

<h2 id="related-work-and-open-questions">Related work and open questions</h2>

<h3>Methods that mix SFT and RL</h3>

Several post-training methods combine SFT and RL within a single run instead of applying them in sequence, and report gains over SFT followed by RL. Their setting differs from ours in two respects. First, the SFT data are solutions generated by a much stronger model, usually DeepSeek-R1, and used to train a 3B–8B model. Second, the method selects which questions or tokens receive this supervision, typically the questions on which every RL attempt fails. Our data, in contrast, come from the same source as the rest of training, and we apply no such selection. The ablations in these papers that remove the selection are therefore the closest published comparison to our schedules.

<div class="interleave-table interleave-text-table" markdown="1">

| Method | How SFT and RL are combined | Reported comparison |
| --- | --- | --- |
| [ReLIFT](https://arxiv.org/abs/2506.07527) | During GRPO, questions on which every attempt fails are added to a buffer; once the buffer is full, one SFT step on R1 solutions to those questions | Qwen2.5-Math-7B, six-benchmark average: 52.6 vs 50.7 and 50.8 for SFT then RL (Table 1) |
| [SRFT](https://arxiv.org/abs/2506.19767) | One loss per step: SFT on R1 solutions, weighted by the policy's entropy, plus two RL terms | Qwen2.5-Math-7B, five math benchmarks: 59.5 vs 54.5 for SFT then RL (Table 2) |
| [BRIDGE](https://proceedings.mlr.press/v306/chen26an.html) | SFT and RL gradients combined at every step, with the SFT term reweighted by a meta-learned objective | Qwen3-8B-Base: 49.9 vs 45.5 for SFT then RL and 42.9 for RL only (Table 3) |
| [EvolveSearch](https://aclanthology.org/2025.emnlp-main.663.pdf) | Each round: SFT of the base model on the correct responses collected so far, then GRPO on a new set of questions | Qwen2.5-7B search agent, after three rounds: +2.4 in-domain and +0.5 out-of-domain over RL only with the same questions (Table 1) |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | SFT → RL → SFT on 600k filtered RL responses plus 200k new non-reasoning examples → RL | AIME 2024: 79.8 vs 77.9 for RL only (R1-Zero); no controlled ablation of the alternation |

</div>

<p class="interleave-note">Numbers are as reported by each paper. None of the papers reports variance across seeds.</p>

Without this selection, the reported gains disappear. When ReLIFT inserts the same number of SFT steps at a fixed interval of 8 RL steps, instead of whenever failed questions accumulate, its average falls from 52.6 to 49.1, below the 50.7 and 50.8 of SFT then RL. Filling the buffer with random questions instead of failed ones yields 45.5, and one SFT step after every RL step yields 23.8 ([ReLIFT, §4.3 and Table 4](https://arxiv.org/abs/2506.07527)). BRIDGE reports that summing the SFT and RL losses with fixed weights scores below RL alone, 29.4 vs. 32.2 on Llama 3.2 3B Instruct and 40.1 vs. 42.9 on Qwen3-8B-Base ([BRIDGE, Tables 2 and 3](https://proceedings.mlr.press/v306/chen26an.html)). In BRIDGE's experiments, SRFT also scores below SFT then RL on both models, although SRFT reports a gain for Qwen2.5-Math-7B, and SRFT itself finds that running RL before SFT ends at 37.4, below SFT alone at 47.3 ([SRFT, Table 1](https://arxiv.org/abs/2506.19767)). A theoretical analysis shows that RL increases the SFT loss and SFT decreases the RL reward, so the two stages cannot be optimized independently ([Niu et al., 2026](https://arxiv.org/abs/2601.07389)).

These ablations attribute the published gains to the content of the SFT data and to which questions receive it, not to alternating SFT and RL. Our experiments hold both fixed, since the data are identical across schedules and every example receives the same loss. Under these conditions we find no benefit, and neither do the ablations above.

Two lines of work are closer to our pretraining experiments. RL Excursions applies RL to intermediate pretraining checkpoints but does not resume pretraining afterwards. [RLP](https://arxiv.org/abs/2510.01265) uses an RL objective during pretraining itself, rewarding a sampled chain of thought by how much it increases the likelihood of the next token. Neither paper resumes standard pretraining after an RL stage and compares against the same data and RL budget without the intermediate stage, and to our knowledge the schedules in this post are the first such comparison.

<h3>Open directions</h3>

We tested only one interleaved schedule, in which a single RL stage at the midpoint of pretraining is followed by a single return to pretraining. However, pretraining and RL can be interleaved in many other ways. For example, alternating between the two in many short rounds would shorten each pretraining stage and might reduce how much of the preceding RL gain it erases. Moreover, the two could follow a curriculum, in which RL is applied to each task as soon as the model's pass@k on it becomes nonzero, or in which the pretraining data after each RL stage are selected according to the questions on which RL failed. We leave these schedules to future work.
