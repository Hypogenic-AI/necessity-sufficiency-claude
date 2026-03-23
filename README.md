# Necessity, Sufficiency, and Selectivity in Mechanistic Interpretability

A systematic evaluation of whether model components identified via necessity (ablation) and sufficiency (activation patching) interventions are *selective*—specific to their target behavior or also affecting unrelated behaviors.

## Key Findings

- **29–42% of top-ranked components leak across tasks**: Components identified as important for one task (e.g., IOI) also significantly affect unrelated tasks (e.g., subject-verb agreement). This is a fundamental limitation of single-task evaluations.
- **Necessity and sufficiency give similar rankings**: Spearman ρ = 0.76–0.88 across tasks. Both interventions largely identify the same components, though IOI shows the most divergence (consistent with redundant OR-circuit structure).
- **No significant selectivity difference between intervention types**: Sufficiency shows slightly higher selectivity (+0.019) but this is not statistically significant (p = 0.52, bootstrap 95% CI includes zero).
- **MLP layers are the primary leakage source**: MLP_0 dominates all tasks. MLP_8–11 show high cross-task effects. Attention heads tend to be more task-specific.
- **Task similarity drives overlap**: SV Agreement and Greater-Than share 40–47% of top components; IOI is more distinct (20–33% overlap).

## Reproduction

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install torch transformer-lens numpy scipy matplotlib seaborn pandas datasets

# Run experiments (~70 seconds on GPU)
CUDA_VISIBLE_DEVICES=0 python src/experiment.py

# Run extended analysis
python src/deeper_analysis.py
```

## File Structure

```
├── REPORT.md                  # Full research report with results and analysis
├── planning.md                # Research plan and hypothesis decomposition
├── src/
│   ├── experiment.py          # Main experiment: patching + selectivity
│   └── deeper_analysis.py     # Extended analysis: universality, bootstrap CIs
├── results/
│   ├── summary.json           # Key metrics and statistical tests
│   ├── extended_summary.json  # General-purpose component analysis
│   ├── patching_results.json  # Full patching scores (all 156 components × 3 tasks)
│   ├── selectivity_data.csv   # Cross-task selectivity measurements
│   ├── selectivity_filtered.csv
│   └── plots/
│       ├── *_heatmaps.png     # Necessity/sufficiency heatmaps per task
│       ├── *_nec_vs_suf.png   # Scatter: necessity vs. sufficiency
│       ├── overall_selectivity.png
│       ├── selectivity_filtered_comparison.png
│       ├── component_universality.png
│       ├── bootstrap_selectivity_diff.png
│       ├── per_component_leakage_profiles.png
│       └── cross_task_heatmap.png
├── literature_review.md       # Synthesized review of 9 key papers
├── resources.md               # Catalog of datasets, papers, code
├── papers/                    # 17 downloaded research papers
├── datasets/                  # MIB and BLiMP datasets
└── code/                      # TransformerLens, pyvene, CausalGym, MIB, feature-circuits
```

See [REPORT.md](REPORT.md) for the full research report.
