# Research Plan: Necessity, Sufficiency, and Selectivity in Mechanistic Interpretability

## Motivation & Novelty Assessment

### Why This Research Matters
Mechanistic interpretability aims to identify which model components causally drive specific behaviors. Current practice relies on necessity (ablation) and sufficiency (activation patching) tests, but these tell us whether a component matters for a behavior—not whether it matters *only* for that behavior. Without selectivity, we risk attributing behaviors to polysemantic or general-purpose components, undermining the utility of circuit-level explanations for model editing, safety auditing, and scientific understanding.

### Gap in Existing Work
Mueller et al. (2024) explicitly identify selectivity as "not often explicitly discussed nor measured." CausalGym (Arora et al., 2024) introduces a selectivity metric but only for 1D linear features on linguistic tasks. MIB (2025) evaluates sufficiency (CPR) and necessity (CMD) but not cross-task selectivity. No existing work systematically measures whether components identified via necessity vs. sufficiency interventions differ in their selectivity profiles.

### Our Novel Contribution
We conduct the first systematic cross-task selectivity evaluation of components identified via necessity (ablation) and sufficiency (activation patching) interventions. We measure whether top components for one task also affect unrelated tasks, and compare selectivity profiles between necessity-ranked and sufficiency-ranked components. This reveals whether the choice of intervention type affects the specificity of discovered circuits.

### Experiment Justification
- **Experiment 1 (Component Ranking)**: Rank all attention heads and MLP layers by necessity and sufficiency scores across 3 tasks. Needed to establish which components each method identifies as important.
- **Experiment 2 (Cross-Task Selectivity)**: Test top-K components from each task on all other tasks. Needed to quantify selectivity—the core gap identified in the literature.
- **Experiment 3 (Necessity vs. Sufficiency Divergence)**: Compare rank orderings between necessity and sufficiency. Tests whether these interventions identify the same or different components, and whether they differ in selectivity.

## Research Question
Do components identified as necessary (via ablation) and sufficient (via activation patching) for a specific behavior differ in their selectivity—i.e., their specificity to that behavior vs. unrelated behaviors?

## Hypothesis Decomposition
- H1: Components ranked highly by necessity and sufficiency will show significant overlap but not be identical.
- H2: Components identified via sufficiency (denoising patching) will show higher selectivity than those identified via necessity (noising/ablation), because sufficiency tests a more targeted causal claim.
- H3: Current single-task evaluations overestimate the causal specificity of identified components—many "important" components will also affect unrelated tasks.

## Proposed Methodology

### Approach
Use GPT-2 Small with TransformerLens. For three distinct tasks (IOI, subject-verb agreement, greater-than/arithmetic), perform both noising (necessity) and denoising (sufficiency) activation patching across all attention heads and MLP layers. Then test the top-K components from each task on all other tasks to measure selectivity.

### Experimental Steps
1. Load GPT-2 Small via TransformerLens
2. Prepare datasets: IOI (from MIB), subject-verb agreement (from BLiMP), and a third distinct task
3. For each task: run noising patching (necessity) and denoising patching (sufficiency) across all 144 attention heads and 12 MLP layers
4. Rank components by effect size for each task × intervention type
5. For top-K components per task: measure effect on all other tasks
6. Compute selectivity scores and compare across intervention types
7. Statistical analysis of divergence and selectivity differences

### Baselines
- Random component selection (null baseline for selectivity)
- Mean ablation vs. zero ablation vs. resampling ablation (comparing ablation types)

### Evaluation Metrics
- **Necessity score**: Logit difference drop when component is ablated (noised)
- **Sufficiency score**: Logit difference recovery when component is patched (denoised)
- **Selectivity**: Effect on target task minus mean effect on non-target tasks
- **Rank correlation**: Spearman ρ between necessity and sufficiency rankings
- **Cross-task leakage**: Fraction of top-K components that significantly affect ≥1 non-target task

### Statistical Analysis Plan
- Spearman rank correlation for necessity vs. sufficiency rankings (α=0.05)
- Paired t-test or Wilcoxon signed-rank for selectivity differences
- Bootstrap confidence intervals for selectivity scores
- Multiple comparison correction via Benjamini-Hochberg FDR

## Expected Outcomes
- H1 supported if Spearman ρ is moderate (0.3-0.7) between necessity and sufficiency rankings
- H2 supported if sufficiency-identified components show significantly higher selectivity
- H3 supported if >30% of top-K components show significant effects on non-target tasks

## Timeline and Milestones
1. Environment setup and data prep: 15 min
2. Core patching infrastructure: 30 min
3. Run patching experiments across 3 tasks: 45 min
4. Cross-task selectivity evaluation: 30 min
5. Analysis and visualization: 30 min
6. Documentation: 30 min

## Potential Challenges
- Self-repair/Hydra effect may confound necessity results → document and analyze
- Task difficulty differences may affect absolute scores → use normalized metrics
- Some components may be general-purpose (LayerNorm, embeddings) → focus on attention heads and MLPs

## Success Criteria
- Completed cross-task selectivity measurements for ≥3 tasks
- Quantified divergence between necessity and sufficiency rankings
- Statistical test of selectivity difference between intervention types
- Clear visualization of results
