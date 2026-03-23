# Literature Review: Necessity, Sufficiency, and Selectivity in Mechanistic Interpretability

## Research Area Overview

Mechanistic interpretability (MI) aims to reverse-engineer the internal computations of neural networks by identifying causal relationships between model components and behaviors. The dominant methodology uses **causal interventions** on internal activations to determine which components mediate specific behaviors. Two fundamental intervention types map to the philosophical concepts of necessity and sufficiency:

- **Necessity** (tested via ablation): Does removing/corrupting a component degrade the behavior?
- **Sufficiency** (tested via activation patching): Does restoring/injecting a component's activation recover/produce the behavior?

A critical but under-explored third criterion is **selectivity**: Does the component specifically affect *only* the targeted behavior, or does it also influence unrelated behaviors? Mueller et al. (2024) explicitly identify selectivity as "not often explicitly discussed nor measured" and note that the assumption that maximizing faithfulness and sparsity implicitly optimizes selectivity remains unverified.

## Key Papers

### Paper 1: The Quest for the Right Mediator (Mueller et al., 2024)
- **Authors**: Aaron Mueller, Jannik Brinkmann, Millicent Li, et al.
- **Year**: 2024
- **Source**: arXiv:2408.01416
- **Key Contribution**: Comprehensive survey organizing MI through the lens of causal mediation analysis, taxonomized by **mediator type** (full layers, neurons, attention heads, non-basis-aligned subspaces, non-linear features) rather than by method.
- **Methodology**: Maps Pearl's causal mediation framework to MI. Defines indirect effect (IE) as the core metric. Identifies four evaluation criteria: **sparsity**, **generality**, **selectivity**, **faithfulness**.
- **Key Findings on Necessity/Sufficiency/Selectivity**:
  - Ablations (zero, mean, resampling) test **necessity**: "They allow us to tell whether any of the information in a mediator is necessary for a model to perform the task" -- high recall, low precision
  - Activation patching (input-dependent deterministic) tests **sufficiency**: replacing a component with a contrastive value tests if it suffices to flip behavior
  - **Selectivity is the key undermeasured criterion**: defined as explaining "only the phenomenon of interest and as little else as possible." Neurons are "sparse but not selective" due to polysemanticity. Non-basis-aligned directions offer significant selectivity advantages.
  - Provides decision tree: explaining behavior → SAE features; verifying hypothesis → DAS; localization/editing → exhaustive search over layers/neurons
- **Datasets**: IOI, subject-verb agreement, factual recall, greater-than, gender bias, entity tracking, addition, MCQA, toxicity
- **Code**: References pyvene, TransformerLens, MIB benchmark

### Paper 2: How to Use and Interpret Activation Patching (Heimersheim & Nanda, 2024)
- **Authors**: Stefan Heimersheim, Neel Nanda
- **Year**: 2024
- **Source**: arXiv:2404.15255
- **Key Contribution**: Explicit conceptual framework mapping patching directions to necessity vs. sufficiency. Central insight: **denoising tests sufficiency; noising tests necessity**, and these are NOT symmetric.
- **Key Findings**:
  - **AND circuits** (serial components): Noising finds all necessary components; denoising fails (neither alone is sufficient)
  - **OR circuits** (parallel/redundant): Denoising finds all sufficient components; noising fails (neither alone is necessary due to redundancy)
  - Denoising finds a "cross-section" of the circuit, not the full circuit
  - **Hydra effect/self-repair**: Backup heads compensate when primary components are ablated (~0.7x compensation), complicating necessity claims
  - Logit difference is the recommended metric for selectivity (controls for shared effects)

### Paper 3: Towards Best Practices of Activation Patching (Zhang & Nanda, 2023)
- **Authors**: Fred Zhang, Neel Nanda
- **Year**: 2023
- **Source**: arXiv:2309.16042
- **Key Contribution**: Systematic empirical comparison of corruption methods and metrics across 5 tasks.
- **Key Findings**:
  - Gaussian noising (GN) vs. symmetric token replacement (STR) can lead to inconsistent circuit discovery. GN puts model off-distribution, breaking internal mechanisms. STR preferred.
  - Probability metric fails to detect negative components (floor effect). Logit difference preferred.
  - Which tokens are corrupted determines what information flows are traced
  - Sliding window patching produces 20%+ more peak effect than summing individual layers

### Paper 4: CausalGym (Arora et al., 2024)
- **Authors**: Aryaman Arora, Daniel Jurafsky, Christopher Potts
- **Year**: 2024
- **Source**: arXiv:2402.12560
- **Key Contribution**: Benchmark for comparing causal interpretability methods on 29 linguistic tasks from SyntaxGym. Introduces **selectivity metric** adapted from control tasks.
- **Methods Evaluated**: DAS, linear probing, difference-in-means, LDA, PCA, k-means, random
- **Key Findings on Selectivity**:
  - DAS achieves highest raw causal efficacy but also performs well on arbitrary control tasks
  - After selectivity adjustment, probing exceeds DAS at larger scales (4.24 vs 3.34 at pythia-1b)
  - Selectivity = odds-ratio on real task minus odds-ratio on control task
- **Models**: Pythia family (14M to 6.9B)
- **Code**: github.com/aryamanarora/causalgym

### Paper 5: MIB: A Mechanistic Interpretability Benchmark (Mueller et al., 2025)
- **Authors**: Aaron Mueller, Atticus Geiger, Sarah Wiegreffe, et al.
- **Year**: 2025
- **Source**: arXiv:2504.13151
- **Key Contribution**: Two-track benchmark: circuit localization (finding minimal subgraphs) and causal variable localization (finding feature representations).
- **Tasks**: IOI, Arithmetic, MCQA, ARC, RAVEL
- **Key Findings**:
  - EAP-IG-inputs with counterfactual ablations best for circuit localization
  - DAS consistently best for causal variable localization
  - **SAE features do NOT improve over standard neurons** for causal variable localization -- notable negative result
  - CPR measures sufficiency (circuit alone reproduces behavior); CMD measures necessity (removing circuit degrades behavior)
- **Models**: GPT-2 Small, Qwen-2.5 0.5B, Gemma-2 2B, Llama-3.1 8B
- **Code**: github.com/aaronmueller/MIB

### Paper 6: Sparse Feature Circuits (Marks et al., 2024)
- **Authors**: Samuel Marks, Can Rager, Eric Michaud, et al.
- **Year**: 2024
- **Source**: arXiv:2403.19647
- **Key Contribution**: Discovers circuits using SAE features instead of neurons/attention heads. Introduces SHIFT method for removing task-irrelevant features.
- **Key Findings**:
  - ~100 SAE features explain subject-verb agreement in Pythia-70M (vs ~1500 neurons)
  - Faithfulness (sufficiency) and completeness (necessity) evaluated
  - SHIFT demonstrates selectivity in practice: ablating human-judged irrelevant features removes gender bias while preserving profession classification
- **Code**: github.com/saprmarks/feature-circuits

### Paper 7: Causal Abstraction: A Theoretical Foundation (Geiger et al., 2023)
- **Authors**: Atticus Geiger, Duligur Ibeling, Amir Zur, et al.
- **Year**: 2023
- **Source**: arXiv:2301.04709
- **Key Contribution**: Formal theoretical framework grounding MI in causal abstraction theory. Defines interchange intervention accuracy (IIA) as a measure of counterfactual faithfulness.

### Paper 8: Mechanistic Interpretability for AI Safety -- A Review (Bereska & Gavves, 2024)
- **Authors**: Leonard Bereska, Efstratios Gavves
- **Year**: 2024
- **Source**: arXiv:2404.14082
- **Key Contribution**: Comprehensive review with explicit AND/OR logic framing of necessity vs. sufficiency in activation patching.
- **Key Findings**: Clean-to-corrupted patching tests sufficiency (OR gates); corrupted-to-clean tests necessity (AND gates). Self-repair/Hydra effect confounds necessity claims.

### Paper 9: Open Problems in Mechanistic Interpretability (Sharkey et al., 2025)
- **Authors**: Lee Sharkey et al.
- **Year**: 2025
- **Source**: arXiv:2501.16496
- **Key Contribution**: Identifies SDL (sparse dictionary learning) limitations for selectivity: feature splitting, feature absorption, interpretability illusions.
- **Key Findings**: SDL activations describe but are not mechanisms; sparsity is not a good proxy for interpretability; no canonical causal mediators identified despite formal causal frameworks.

## Common Methodologies

### Intervention Types
- **Ablation** (necessity): Zero, mean, resampling ablation -- Used in Papers 1, 2, 3, 5, 6, 8
- **Activation patching** (sufficiency): Replace activations from contrastive input -- Used in Papers 1, 2, 3, 5, 8
- **Path patching**: Restrict effects to specific downstream paths -- Used in Papers 2, 6
- **DAS/Boundless DAS** (sufficiency via optimization): Learn subspaces -- Used in Papers 1, 4, 5, 7
- **Attribution patching** (gradient approximation): Fast O(1) approximation -- Used in Papers 5, 6

### Circuit Discovery Methods
- Exhaustive activation patching over components
- Attribution patching (gradient-based approximation)
- Edge attribution patching with integrated gradients (EAP-IG)
- SAE feature circuit discovery

## Standard Baselines
- Random features/circuits (lower bound)
- Full vector intervention (upper bound for causal variable localization)
- Difference-in-means (simple but competitive)
- Linear probing (correlational, not causal)
- DAS (current state-of-art for causal variable localization)

## Evaluation Metrics
- **Faithfulness/CPR**: Does the circuit/feature reproduce the behavior? (sufficiency)
- **Completeness/CMD**: Does the complement fail to reproduce? (necessity)
- **IIA**: Interchange intervention accuracy (causal faithfulness)
- **Logit difference**: Preferred metric -- controls for shared effects, linear in residual stream
- **Selectivity**: Real task performance minus control task performance (CausalGym)
- **Log odds-ratio**: CausalGym's primary causal efficacy metric

## Gaps and Opportunities

1. **Selectivity is systematically under-measured**: Mueller et al. (2024) explicitly states this. Only CausalGym (2024) introduces a selectivity metric, and only for 1D linear features.

2. **Necessity ≠ ¬Sufficiency**: Heimersheim & Nanda (2024) show these give different results depending on circuit structure (AND vs OR gates). No systematic framework quantifies the gap.

3. **Self-repair/Hydra effect undermines necessity claims**: When components are ablated, backup mechanisms compensate. This is documented but not systematically quantified across tasks.

4. **SAE features may not improve selectivity**: MIB (2025) shows SAE features perform no better than neurons for causal variable localization. This challenges a core assumption of the feature-based interpretability paradigm.

5. **No cross-task selectivity benchmark exists**: Current benchmarks evaluate methods on individual tasks. No benchmark systematically measures whether identified components are specific to one task or involved in many.

## Recommendations for Our Experiment

Based on the literature review:

- **Recommended datasets**: MIB benchmark tasks (IOI, Arithmetic, MCQA) with counterfactual pairs; CausalGym linguistic tasks for selectivity evaluation; BLiMP for additional linguistic coverage
- **Recommended baselines**: DAS, difference-in-means, linear probing, attribution patching (EAP-IG), random features
- **Recommended metrics**: Faithfulness (sufficiency), completeness (necessity), selectivity (cross-task specificity), logit difference as primary output metric
- **Recommended models**: GPT-2 Small (most studied, fastest), Pythia-70M (SAE feature circuits available), Gemma-2 2B (SAE coverage via GemmaScope)
- **Methodological considerations**:
  - Use symmetric token replacement over Gaussian noise for corruption
  - Test both patching directions (denoising for sufficiency, noising for necessity)
  - Measure selectivity explicitly by testing identified components on unrelated tasks
  - Use logit difference as primary metric, report probability and KL divergence as secondary
  - Account for self-repair effects when interpreting necessity results
