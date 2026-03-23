# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project on "Necessity, Sufficiency, and Selectivity in Mechanistic Interpretability." The research hypothesis is that current MI interventions insufficiently address selectivity -- whether a component specifically affects only the targeted behavior -- and that improving definitions and evaluations of necessity and sufficiency to better capture selectivity will clarify the causal role of components in model behaviors.

## Papers
Total papers downloaded: 17

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| The Quest for the Right Mediator | Mueller et al. | 2024 | papers/mueller2024_quest_right_mediator.pdf | Central survey; selectivity as undermeasured criterion |
| How to Use and Interpret Activation Patching | Heimersheim & Nanda | 2024 | papers/heimersheim2024_activation_patching_interpret.pdf | Denoising=sufficiency, noising=necessity framework |
| Best Practices of Activation Patching | Zhang & Nanda | 2023 | papers/zhang2023_activation_patching_best_practices.pdf | STR vs GN; metric comparison |
| CausalGym | Arora et al. | 2024 | papers/arora2024_causalgym.pdf | Selectivity benchmark with control tasks |
| MIB Benchmark | Mueller et al. | 2025 | papers/mueller2025_mib_benchmark.pdf | CPR/CMD metrics; SAEs vs neurons |
| Sparse Feature Circuits | Marks et al. | 2024 | papers/marks2024_sparse_feature_circuits.pdf | SAE circuit discovery; SHIFT method |
| Locating and Editing Factual Associations (ROME) | Meng et al. | 2022 | papers/meng2022_rome_locating_editing.pdf | Pioneered causal tracing |
| IOI Circuit | Wang et al. | 2022 | papers/wang2022_ioi_interpretability_wild.pdf | Canonical circuit analysis |
| Causal Abstractions of Neural Networks | Geiger et al. | 2021 | papers/geiger2021_causal_abstractions.pdf | Foundational causal abstraction |
| Causal Abstraction: Theoretical Foundation | Geiger et al. | 2023 | papers/geiger2023_causal_abstraction_foundation.pdf | Extended theoretical framework |
| Automated Circuit Discovery (ACDC) | Conmy et al. | 2023 | papers/conmy2023_automated_circuit_discovery.pdf | Automated circuit finding |
| Progress Measures for Grokking | Nanda et al. | 2023 | papers/nanda2023_grokking_mechanistic.pdf | Circuit analysis methodology |
| MI for AI Safety Review | Bereska & Gavves | 2024 | papers/bereska2024_mech_interp_safety_review.pdf | AND/OR gate framing |
| Open Problems in MI | Sharkey et al. | 2025 | papers/sharkey2025_open_problems_mech_interp.pdf | SDL limitations; feature splitting |
| SAEs Find Interpretable Features | Cunningham et al. | 2023 | papers/cunningham2023_sae_interpretable_features.pdf | Foundational SAE work |
| AtP* | Kramar et al. | 2024 | papers/kramar2024_atp_star.pdf | Efficient attribution patching |
| Invariant Algorithmic Cores | Schiffman | 2026 | papers/schiffman2026_invariant_algorithmic_cores.pdf | Necessary and sufficient subspaces |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 6

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| MIB IOI | mib-bench/ioi | 21K examples | Indirect object identification | datasets/mib_ioi/ | 8 counterfactual types |
| MIB Arithmetic | mib-bench/arithmetic_addition | 40K examples | Two-digit addition | datasets/mib_arithmetic_addition/ | 7 counterfactual types |
| MIB MCQA | mib-bench/copycolors_mcqa | 210 examples | Object color QA | datasets/mib_mcqa/ | 9 counterfactual types |
| BLiMP Anaphor Agreement | nyu-mll/blimp | 1K pairs | Anaphor number agreement | datasets/blimp_anaphor_number_agreement/ | Minimal pairs |
| BLiMP SV Agreement | nyu-mll/blimp | 1K pairs | Subject-verb agreement | datasets/blimp_regular_plural_subject_verb_agreement_1/ | Minimal pairs |
| BLiMP Distractor Agreement | nyu-mll/blimp | 1K pairs | Agreement across relative clauses | datasets/blimp_distractor_agreement_relative_clause/ | Minimal pairs |

See datasets/README.md for download instructions and detailed descriptions.

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | MI library with HookPoints | code/TransformerLens/ | Core tool for ablation/patching |
| pyvene | github.com/stanfordnlp/pyvene | Causal intervention library | code/pyvene/ | DAS, interchange interventions |
| CausalGym | github.com/aryamanarora/causalgym | Selectivity benchmark | code/causalgym/ | 29 tasks, selectivity metric |
| Feature Circuits | github.com/saprmarks/feature-circuits | SAE circuit discovery | code/feature-circuits/ | SHIFT, faithfulness/completeness |
| MIB | github.com/aaronmueller/MIB | MI benchmark | code/MIB/ | CPR/CMD metrics, leaderboard |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service for initial broad search (83 results)
2. Queried Semantic Scholar API for targeted searches on necessity/sufficiency/selectivity
3. Identified specific seminal papers via domain knowledge and citation chains
4. Cross-referenced datasets and code from paper methods sections

### Selection Criteria
- Papers directly addressing necessity, sufficiency, or selectivity in MI
- Benchmark papers providing evaluation frameworks for these concepts
- Foundational papers establishing methods (activation patching, causal abstraction, SAEs)
- Review papers synthesizing the field and identifying gaps
- Code repositories actively maintained and used in recent publications

### Challenges Encountered
- Semantic Scholar API rate limiting required delays between queries
- Some key works (Towards Monosemanticity, Mathematical Framework for Transformer Circuits, Causal Scrubbing) published only on blogs/forums without arXiv PDFs
- BLiMP dataset requires per-config loading (67 configs)

### Gaps and Workarounds
- Anthropic Transformer Circuits publications lack arXiv PDFs; referenced via URLs in literature review
- Causal Scrubbing (Redwood Research) published on Alignment Forum only; described from secondary sources
- No pre-existing cross-task selectivity benchmark exists; this is a key gap our research addresses

## Recommendations for Experiment Design

Based on gathered resources, recommend:

1. **Primary dataset(s)**: MIB IOI and MIB Arithmetic -- well-studied tasks with counterfactual pairs, known circuits, and standardized evaluation. BLiMP subsets for cross-task selectivity testing.

2. **Baseline methods**:
   - Activation patching (both directions: denoising for sufficiency, noising for necessity)
   - Attribution patching (EAP-IG) for efficient approximation
   - DAS for causal variable localization
   - Difference-in-means as simple baseline
   - Random features as lower bound

3. **Evaluation metrics**:
   - **Faithfulness/CPR** (sufficiency): Does the circuit alone reproduce behavior?
   - **Completeness/CMD** (necessity): Does removing the circuit degrade behavior?
   - **Selectivity** (novel): Does the circuit affect ONLY the targeted behavior? Measure using cross-task evaluation or CausalGym-style control tasks.
   - **Logit difference** as primary output metric

4. **Code to adapt/reuse**:
   - TransformerLens for model loading and hook-based interventions
   - pyvene for DAS and causal abstraction experiments
   - CausalGym's selectivity metric implementation
   - MIB's CPR/CMD evaluation code
   - Feature Circuits for SAE-based analysis

5. **Recommended models**: GPT-2 Small (fastest, most studied), Pythia-70M (SAE circuits available)

6. **Experimental design suggestion**: For each task (IOI, arithmetic, agreement):
   - Identify top-K components via activation patching (sufficiency) and ablation (necessity)
   - Measure overlap and divergence between necessity and sufficiency rankings
   - Test identified components on unrelated tasks to measure selectivity
   - Compare selectivity across intervention types and component granularities
