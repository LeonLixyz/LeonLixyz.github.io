---
layout: distill
title: Language Models Can Think, But Not Learn
date: 2026-06-15
description: Language models can reason over what is in the context window, but they do not have a reliable way to turn new data or new experience into lasting memory.
authors:
  - name: Leon (Ang) Li
    url: https://leonlixyz.github.io/
    affiliations:
      name: New York University
      url: https://www.nyu.edu/
custom_stylesheets:
  - /assets/css/learning-agent.css
toc:
  - name: What is learning
  - name: Long context is not the solution
  - name: How brains do it, and what we have tried
  - name: Towards real self-improving agents
---

<!-- Migrated from LeonLixyz/learning-agent-blog/build/assemble.py. Edit this post and its figure includes directly. -->
<p>In the movie Memento, the main character cannot form new memories. He has a conversation, solves a problem, learns a name, and minutes later it is all gone. To cope, he covers himself in tattoos and Polaroids and re-reads them every morning, just to work out who he is and what he is doing.</p>

<p>This is exactly where a language model is today. After pretraining and post-training its weights are frozen: the durable store of everything it knows stops changing. Whatever it works out during a session lives only in the context window, its working memory, and the moment the session ends that is wiped. Inside that window it can <em>think</em>, but it cannot <em>learn</em>: nothing it figures out from new data or new experience becomes part of the model. As Richard Sutton puts it, deployed models spend almost all of their compute not learning from the experience they are having (<a href="https://www.dwarkesh.com/p/richard-sutton">Sutton, 2025</a>).</p>

{% include learning-agent/memory-map.liquid %}

<h2 id="what-is-learning">What is learning</h2>
<h3 id="sec-1-1">1.1 What learning is</h3>

<p>Learning is not the same as having information nearby. A database has information. A context window has information. A learner is different: it turns experience into compressed, reusable knowledge and skills. If you read a proof, practice a serve, or get sick from a food, the point is not that you can replay the episode or recite the fact, but that later, in a situation similar but not the same, you can use what you took from it. That is generalization, and it is what learning is for.</p>

<p>Why not just write the experience straight into a language model's weights with backpropagation? The catch is how knowledge forms. A model turns text into reliably stored, extractable knowledge only after it has seen a fact many times during training. Allen-Zhu and Li find that reaching a model's full storage capacity, roughly two bits per parameter, takes on the order of a thousand exposures per fact, and that training on ten times fewer exposures roughly halves what it retains (<a href="https://arxiv.org/abs/2404.05405">Allen-Zhu &amp; Li, 2024</a>). A corpus that states something once sits far below that, so the practical fix is to synthesize many diverse restatements and train on those (<a href="https://arxiv.org/abs/2409.07431">Yang et al., 2024</a>). This is the <strong>curse of learning</strong>: turning a single observation into knowledge the model can use and generalize from takes on the order of a thousand varied repetitions, while your source gives you one.</p>

<h2 id="long-context-is-not-the-solution">Long context is not the solution</h2>
<h3 id="sec-2-1">2.1 Long context as working memory</h3>

<p>People have tried plenty of ways around the curse, and the most obvious is to not write to the weights at all: just keep everything in the context window or an external database. Context, the model's working memory, has a famous name: <strong>in-context learning</strong>. The name is misleading. Given a few examples in the prompt, a model gets better at the task without any training. That is impressive, but it is mostly <strong>inference</strong>: the model thinks with the evidence in front of it. The "update" is a change in the prompt, the cache, or the temporary state of the run, not a change in the durable learner.</p>

<p>But why not just make the window enormous and pour everything in? Three reasons.</p>

<ul>
<li><strong>Data.</strong> To use a 100M-token window the model must be trained on tasks that truly need 100M-token integration, and that data barely exists, so effective context lags far behind the advertised number (<a href="https://arxiv.org/abs/2307.03172">Liu et al., 2023</a>; <a href="https://arxiv.org/abs/2404.06654">Hsieh et al., 2024</a>).</li>
<li><strong>Compute.</strong> Every query re-reads the whole history, and attention's cost grows with the square of the context length, so past some point the window is simply too expensive to run.</li>
<li><strong>No abstraction.</strong> A system that never compresses never builds skills; it re-reads the whole context and works each problem from scratch instead of getting better over time.</li>
</ul>

<p>There's a whole industry making that working memory go further: building external memory systems, compacting the conversation, shrinking the KV cache by eviction or quantization. These genuinely help: they stretch how long a context can run, which matters for long-document inference and for long-horizon <em>agents</em> that would otherwise overflow. All of them optimize the same thing: the <em>size and reach</em> of the working memory. They improve the conditions for thinking. They do not create a durable write. However much the context is compressed, it is gone when the session ends. Bigger, cheaper RAM is still RAM. The same applies to long-context-style updates: if the update lives only in the run, it is a better thought, not a learned change.</p>

<h3 id="sec-2-2">2.2 Linear attention compresses and decays in working memory</h3>

<p>The first architectural "fix" trades attention's exact-but-expensive log for a <em>fixed-size</em> summary. State-space models like Mamba keep a running state $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$, learning to selectively remember and forget (<a href="https://arxiv.org/abs/2312.00752">Gu &amp; Dao, 2023</a>). Constant memory per token, runs forever, but the summary is lossy and must overwrite old information. Gu calls it "database vs. brain." It's a hard Pareto tradeoff: recall is bounded by state size (<a href="https://arxiv.org/abs/2402.18668">Arora et al., 2024</a>).</p>

<p>Forgetting itself is not the problem; selective forgetting is what makes abstraction possible. Borges' Funes the Memorious remembers everything and is, as a result, incapable of thought: he cannot see why "dog" should cover so many different animals. Abstraction is forgetting on purpose, and the slow cortex earns its concepts by dropping particulars. The trouble is that today's architectures forget in <strong>exactly the wrong place</strong>. Mamba, linear attention, TTT, even Titans bake decay into the <em>working-memory</em> state, which confuses working memory with knowledge. Working memory should not forget at all; it should let the model look back at anything it has seen, exactly, and on that axis <strong>attention is right</strong>. Forgetting belongs in the slow consolidation into long-term knowledge, not in the working-memory buffer.</p>

{% include learning-agent/forgetting.liquid %}

<h3 id="sec-2-3">2.3 Test-time training is just linear attention</h3>

<p>The most interesting "fix": make the working-memory state itself a small model, and read by <em>training</em> it. Test-time training (TTT) takes a gradient step on that state for every token, $W_t = W_{t-1} - \eta\,\nabla_W\,\ell(W_{t-1};x_t)$ (<a href="https://arxiv.org/abs/2407.04620">Sun et al., 2024</a>). It sounds like the model is finally learning as it reads. It is not.</p>

<p>The point is not the exact algebra. It is that test-time training is basically just another form of linear attention: in the linear case, TTT-Linear is literally <strong>DeltaNet</strong>, a delta-rule linear attention model. Calling it "training" makes it sound like real learning, but write the update out, name every piece, and the magic disappears.</p>

<div class="deriv">
  <p><strong>First, the pieces.</strong> Each token is projected into a <em>key</em> $k_t$ and a <em>value</em> $v_t$, exactly as attention makes keys and values: the key says what the token is about, the value is what to return for it. The memory is a matrix $S$ that maps a key to a predicted value, $\hat v = Sk$. To store the association "key $k_t$ returns value $v_t$," you add the <em>outer product</em> $v_tk_t^\top$ to $S$, because that is the matrix which, fed $k_t$, gives back $v_t$. And $\beta_t$ is just how hard you write this token, a per-step learning rate.</p>
  <p><strong>DeltaNet</strong> does not write blindly. For the current key it first reads what the memory already returns, $\hat v_t=S_{t-1}k_t$, takes the error $v_t-\hat v_t$, and writes only that error back along the key:</p>
  <p>$$S_t=S_{t-1}+\beta_t\,(v_t-S_{t-1}k_t)\,k_t^\top.$$</p>
  <p><strong>TTT-Linear</strong> calls the same matrix $W$ and treats it as a tiny linear model $f_W(k)=Wk$. It trains $f$ on the single target $v_t$ with squared loss $\ell_t(W)=\tfrac12\lVert Wk_t-v_t\rVert^2$. The gradient is $\nabla_W\ell_t=(Wk_t-v_t)k_t^\top$, so one gradient step is:</p>
  <p>$$W_t=W_{t-1}-\eta\,\nabla_W\ell_t=W_{t-1}+\eta\,(v_t-W_{t-1}k_t)\,k_t^\top.$$</p>
  <p>Rename $W$ to $S$ and the learning rate $\eta$ to $\beta_t$, and the two lines are identical: the same rank-one write of the same prediction error along the same key. The only difference is the story, recurrent context state versus fast weights.</p>
</div>

{% include learning-agent/deltanet-ttt.liquid %}

<p>DeltaNet and TTT-Linear are just two entries in a whole family of recurrent-update models that trade attention's growing cache for a fixed-size state (<a href="https://arxiv.org/abs/2312.00752">Gu &amp; Dao, 2023</a>; <a href="https://arxiv.org/abs/2405.21060">Dao &amp; Gu, 2024</a>; <a href="https://arxiv.org/abs/2510.26692">Kimi Team, 2025</a>; <a href="https://arxiv.org/abs/2605.22791">Hatamizadeh et al., 2026</a>). They differ mostly in how they erase and write the state, as the table below lays out.</p>

{% include learning-agent/update-family.liquid %}

<p>TTT actually runs gradient <em>training</em> at test time, and it still does not solve learning. Since TTT-Linear is exactly DeltaNet, every limit TTT hits, DeltaNet hits too:</p>

<ul>
<li><strong>The state usually resets.</strong> It is re-initialized every sequence and never folded back into the base model, so the training is thrown away when the sequence ends.</li>
<li><strong>Even a real gradient write is slow to make safe.</strong> A fast write into shared weights overwrites neighboring knowledge (the catastrophic interference of §6), so a safe update has to be slow.</li>
<li><strong>Fixed-state models give up attention's exact recall</strong> to make long context cheaper. Richer variants like TTT-MLP and TTT-E2E push the write further (<a href="https://arxiv.org/abs/2512.23675">Tandon et al., 2025</a>), but the update is still context-local unless something saves it.</li>
</ul>

<p>Calling the update training does not make it learning. These are useful long-context mechanisms, not a fix for the missing write path.</p>

{% include learning-agent/recall.liquid %}

<h3 id="sec-2-4">2.4 The agentic hack</h3>

<p>The other "fix" is to optimize everything <em>around</em> a frozen model, escalating in cleverness: prompt engineering → retrieval (RAG) → context compaction → agentic scaffolds → and the recursive end, where the system optimizes its own harness and even evolves its own scaffold. The escalation <em>is</em> the point: it makes a better inferencer out of the same frozen learner. Even when an agent rewrites the agent that writes its own scaffold, it never touches the weights.</p>

<p>It rests on a quiet category error: it treats context as the model's <strong>memory</strong>, when context is really the model's <strong>senses</strong>. The tell is compaction: when the window fills, you summarize and discard, a lossy digest that nothing consolidates. This is the <em>Memento</em> problem from the opening of this essay. The notes can be brilliant, but the agent still wakes each session no smarter than before. Context optimization, long-context methods, RAG, and long-context-style updates will keep getting useful, but they fail as a theory of learning for the same reason: they sharpen inference while leaving memory unwritten. The harness is the tattoos, a remarkable way to cope with amnesia, not a cure.</p>

<h2 id="how-brains-do-it-and-what-we-have-tried">How brains do it, and what we have tried</h2>
<h3 id="sec-3-1">3.1 Complementary learning systems: two stores and replay</h3>

<p>Brains face our exact problem (learn fast without overwriting everything), and the answer is well developed: <strong>Complementary Learning Systems</strong> (<a href="https://stanford.edu/~jlmcc/papers/McClellandMcNaughtonOReilly95.pdf">McClelland, McNaughton &amp; O'Reilly, 1995</a>; <a href="https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(16)30043-2">Kumaran, Hassabis &amp; McClelland, 2016</a>). Two systems with opposite settings: a <strong>hippocampus</strong> that records episodes fast, one-shot, kept separate; and a <strong>neocortex</strong> that integrates slowly over many exposures into structured knowledge.</p>

<p>Why two? If you write fast and hard into a dense, distributed memory, it bleeds into everything nearby, <strong>catastrophic interference</strong> (McCloskey &amp; Cohen, 1989; French, 1999). The cortex avoids it by learning slowly and interleaved. The slowness is a feature. It's what protects old knowledge. (Same slowness as the thousand-exposures result. Now you know why.) But slow learning alone couldn't remember this morning, so the hippocampus grabs the episode now and, during rest and sleep, <strong>replays</strong> it to the cortex, interleaved, so it consolidates safely.</p>

<p>Mapped onto an LLM, the neocortex is the only part with an analog, and even it has stopped learning:</p>

<ul>
<li>The <strong>weights are the neocortex</strong>: general knowledge, slowly consolidated over many exposures during training. But a neocortex keeps consolidating for life, while the weights were frozen the day training ended and never updated since. We have the store, not the slow learner.</li>
<li><strong>No hippocampus</strong>: no fast, durable, one-shot store. The context window looks like one, but it is wiped when the session ends, so it is scratch paper, not memory.</li>
<li><strong>No replay</strong>: nothing that consolidates the day's experience into the weights.</li>
</ul>

<p>The model has a frozen store and a whiteboard wiped between meetings. That is the gap. The good news: brains prove fast, durable, one-shot writing is possible and need <em>not</em> be a slow gradient grind. The hippocampus writes by association in one shot, much closer to a Hopfield network (<a href="https://arxiv.org/abs/2008.02217">Ramsauer et al., 2020</a>) than to a thousand-step optimization.</p>

{% include learning-agent/cls.liquid %}

<h3 id="sec-3-2">3.2 The primitive attempts</h3>

<p>If we wanted a model that learns, it would need three capabilities. None is solved, but each has a serious 2025–26 attempt, and how each falls short tells you what's hard.</p>

<ul>
<li><strong>Write new knowledge.</strong> Model-editing methods such as MEMIT (<a href="https://arxiv.org/abs/2210.07229">Meng et al., 2022</a>) can patch a fact, and SEAL-style systems (<a href="https://arxiv.org/abs/2506.10943">Zweiger et al., 2025</a>) make a model generate its own training data before updating. The hard part is propagation: the fact should affect related answers without causing drift or forgetting.</li>
<li><strong>Internalize experience.</strong> STaR-style bootstrapping (<a href="https://arxiv.org/abs/2203.14465">Zelikman et al., 2022</a>) turns solved examples into training data, but it depends on a verifier and usually improves only inside a bounded task.</li>
<li><strong>Keep long-term memory.</strong> Titans-like memory modules (<a href="https://arxiv.org/abs/2501.00663">Behrouz et al., 2025</a>) and external memory systems carry more, but most are still better working memory, not durable self-modification.</li>
</ul>

<p>All of this needs a <strong>signal</strong>: what is worth keeping, and how good an attempt was. That, plus a store for experience and a write method, is the toolbox the next section specifies.</p>

<h2 id="towards-real-self-improving-agents">Towards real self-improving agents</h2>
<h3 id="sec-4-1">4.1 The immediate hack: learning as a toolbox</h3>

<p>We have endless data about the world, but almost none about <em>how to learn</em>. Today learning is a fixed recipe: pretraining, then SFT, then RL, the same pipeline for every model and task. People do not learn on a fixed recipe. You imitate, practice, ask a tutor, reflect, and above all <em>consolidate</em>, replaying and recasting an experience until it generalizes, switching between these by what the task allows. Learning should be a <strong>tool the model picks up</strong>, not a rule it is stuck with.</p>

{% include learning-agent/toolbox.liquid %}

<p>The toolbox holds three tools.</p>

<ul>
<li>The <strong>signal</strong> decides what to learn and scores an attempt: evals, verifiers, and environment returns externally; a learned value function or critic internally. External signal is sparse but reliable, internal signal dense but gameable.</li>
<li>The <strong>data tool</strong> holds experience and synthesizes the training set, the consolidation step. The curse is paid here: one observation is expanded into the many representations retention requires, paraphrases, QA pairs, cloze deletions, contrastive negatives, counterexamples, tool-use traces, schema links, eval probes. EntiGraph (synthetic continued pretraining) and SEAL (self-edits) are instances (<a href="https://arxiv.org/abs/2409.07431">Yang et al., 2024</a>; <a href="https://arxiv.org/abs/2506.10943">Zweiger et al., 2025</a>).</li>
<li>The <strong>optimization tool</strong> writes to the weights through an interchangeable method, SFT, RL, on-policy distillation, self-distillation, or a localized weight edit, selected by the signal: dense reward to RL, a reference trajectory to SFT, a graded rollout to distillation. Making the write method a runtime choice rather than a fixed pipeline is most of what separates a trained model from a learning one.</li>
</ul>

<h3 id="sec-4-2">4.2 From wrapped LLM to native learner</h3>

<p>The range has two ends. At one, what we have now: a frozen LLM wrapped in agents and scaffolding. At the other, a native learner rebuilt from scratch, with all four primitives built in, a one-shot hippocampus, a consolidation loop that runs during "sleep," exploration, persistent memory. That is the principled end, and almost certainly where it lands, but brutally hard: redesign the stack, retrain from scratch, and beat a paradigm with a decade of optimization behind it. Every piece is an open problem.</p>

{% include learning-agent/tworoads.liquid %}

<p>The <strong>learning agent</strong> sits in between, and it is what can be built today: the frozen transformer wrapped in a loop that generates data, fine-tunes, checks, and repeats, training on data it produces itself rather than data from a larger teacher (the shift SEAL points at). A hack, bolting memory on from outside, but a working one. Two objections bite here. The Bitter Lesson says just scale, but continual learning is the half of it we have not scaled. And periodic retraining suffices only until the model must adapt to one user, now, from a handful of examples. The whole range gets explored, and the learning agent in the middle is the prototype for the right end.</p>

<h3 id="sec-4-3">4.3 How to train the learning agent</h3>

<p>One question is left: who tunes the agent itself? You should not hand-design the recipe, which data to generate, which method to use, what to keep, any more than you hand-design the knowledge. Let the agent search for it, the way evolutionary methods search: propose a variation, evaluate it, keep what works, archive it, repeat. AlphaEvolve, FunSearch, the Darwin Gödel Machine, and Hyperagents show this propose-evaluate-archive loop genuinely discovers: AlphaEvolve even found a way to multiply 4×4 complex matrices in 48 scalar multiplications, beating Strassen's 49 (<a href="https://arxiv.org/abs/2506.13131">Novikov et al., 2025</a>). Aimed at the learning process itself, that loop lets the agent learn how to learn: it evolves better recipes for turning experience into durable skill, instead of running one fixed recipe forever.</p>

{% include learning-agent/evolution.liquid %}

<hr/>
<h2 id="refs-h">References</h2>
<div class="refs"><ol>
<li>Geva et al. (2021), <em>Transformer Feed-Forward Layers Are Key-Value Memories.</em> arXiv:2012.14913</li>
<li>Lample et al. (2019), <em>Large Memory Layers with Product Keys.</em> arXiv:1907.05242</li>
<li>Berges et al. (2024), <em>Memory Layers at Scale.</em> arXiv:2412.09764</li>
<li>Allen-Zhu &amp; Li (2024), <em>Physics of Language Models 3.3: Knowledge Capacity Scaling Laws.</em> arXiv:2404.05405</li>
<li>Morris et al. (2025), <em>How Much Do Language Models Memorize?</em> arXiv:2505.24832</li>
<li>von Oswald et al. (2023), <em>Transformers Learn In-Context by Gradient Descent.</em> arXiv:2212.07677</li>
<li>Shen et al. (2023), <em>Do Pretrained Transformers Really Learn In-Context by Gradient Descent?</em> arXiv:2310.08540</li>
<li>Hendel et al. (2023), <em>In-Context Learning Creates Task Vectors.</em> arXiv:2310.15916</li>
<li>Liu et al. (2023), <em>Lost in the Middle.</em> arXiv:2307.03172</li>
<li>Hsieh et al. (2024), <em>RULER.</em> arXiv:2404.06654</li>
<li>Eyuboglu et al. (2025), <em>Cartridges.</em> arXiv:2506.06266</li>
<li>Gu &amp; Dao (2023), <em>Mamba.</em> arXiv:2312.00752</li>
<li>Dao &amp; Gu (2024), <em>Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.</em> arXiv:2405.21060</li>
<li>Arora et al. (2024), <em>Based.</em> arXiv:2402.18668</li>
<li>Sun et al. (2024), <em>Learning to (Learn at Test Time).</em> arXiv:2407.04620</li>
<li>Yang, Kautz &amp; Hatamizadeh (2024/2025), <em>Gated Delta Networks: Improving Mamba2 with Delta Rule.</em> arXiv:2412.06464; Kimi Team (2025), <em>Kimi Linear.</em> arXiv:2510.26692; Hatamizadeh, Choi &amp; Kautz (2026), <em>Gated DeltaNet-2.</em> arXiv:2605.22791.</li>
<li>Tandon et al. (2025), <em>End-to-End Test-Time Training for Long Context.</em> arXiv:2512.23675</li>
<li>Ba, Hinton et al. (2016), <em>Using Fast Weights to Attend to the Recent Past.</em> arXiv:1610.06258; Hinton &amp; Plaut (1987); Schmidhuber (1992).</li>
<li>Schlag et al. (2021), <em>Linear Transformers Are Secretly Fast Weight Programmers.</em> arXiv:2102.11174</li>
<li>Wang et al. (2025), <em>Test-Time Regression.</em> arXiv:2501.12352</li>
<li>Bailey, Kandel &amp; Harris (2015), <em>Structural Components of Synaptic Plasticity and Memory Consolidation.</em> Cold Spring Harbor Perspectives in Biology; Kandel (2009), <em>The Biology of Memory: A Forty-Year Perspective.</em> Journal of Neuroscience.</li>
<li>Schultz, Dayan &amp; Montague (1997), <em>A Neural Substrate of Prediction and Reward.</em> Science.</li>
<li>McCloskey &amp; Cohen (1989), <em>Catastrophic Interference in Connectionist Networks</em>; McClelland, McNaughton &amp; O'Reilly (1995); Kumaran, Hassabis &amp; McClelland (2016), Complementary Learning Systems.</li>
<li>Tse et al. (2007), <em>Schemas and Memory Consolidation.</em> Science.</li>
<li>Ramsauer et al. (2020), <em>Hopfield Networks Is All You Need.</em> arXiv:2008.02217</li>
<li>Zweiger et al. (2025), <em>SEAL.</em> arXiv:2506.10943; Meng et al. (2022), <em>MEMIT.</em> arXiv:2210.07229; Hase et al. (2023), arXiv:2301.04213.</li>
<li>Zelikman et al. (2022), <em>STaR.</em> arXiv:2203.14465; Yue et al. (2025), arXiv:2504.13837.</li>
<li>Burda et al. (2018), <em>RND.</em> arXiv:1810.12894; Kirk et al. (2023), arXiv:2310.06452.</li>
<li>Madaan et al. (2023), <em>Self-Refine: Iterative Refinement with Self-Feedback.</em> arXiv:2303.17651</li>
<li>Behrouz et al. (2025), <em>Titans.</em> arXiv:2501.00663; <em>Nested Learning.</em> arXiv:2512.24695.</li>
<li>Novikov et al. (2025), <em>AlphaEvolve.</em> arXiv:2506.13131; Romera-Paredes et al. (2024), <em>FunSearch</em> (Nature); Zhang et al. (2025), <em>Darwin Gödel Machine.</em> arXiv:2505.22954.</li>
<li>Yang et al. (2024), <em>Synthetic Continued Pretraining (EntiGraph).</em> arXiv:2409.07431.</li>
<li>Luo et al. (2023), arXiv:2308.08747; Villalobos et al. (2024), arXiv:2211.04325.</li>
<li>Sutton (2019), <em>The Bitter Lesson</em>; Sutton (2025), Dwarkesh Podcast; Silver &amp; Sutton (2025), <em>The Era of Experience</em>; Borges (1942), <em>Funes the Memorious.</em></li>
<li>Mitchell (1997), <em>Machine Learning.</em> McGraw-Hill (the experience / task / performance definition, p. 2).</li>
<li>Hebb (1949), <em>The Organization of Behavior.</em> Wiley (the neurophysiological postulate, p. 62).</li>
<li>Hassabis (2026), interview with Alex Kantrowitz, <em>Big Technology.</em></li>
</ol></div>

<script defer src="{{ "/assets/js/learning-agent.js" | relative_url | bust_file_cache }}"></script>
