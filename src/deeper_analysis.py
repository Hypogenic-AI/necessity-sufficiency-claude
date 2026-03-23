"""
Deeper analysis of selectivity results:
1. Examine MLP_0 dominance and general-purpose components
2. Exclude general-purpose components and re-analyze selectivity
3. Bootstrap confidence intervals for selectivity
4. Detailed cross-task leakage patterns
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"

# Load results
with open(f"{RESULTS_DIR}/patching_results.json") as f:
    all_results = json.load(f)
with open(f"{RESULTS_DIR}/summary.json") as f:
    summary = json.load(f)
selectivity_df = pd.read_csv(f"{RESULTS_DIR}/selectivity_data.csv")

task_names = ["IOI", "SV_Agreement", "Greater_Than"]

# ============================================================
# 1. Identify general-purpose components (high effect on ALL tasks)
# ============================================================
print("=" * 60)
print("General-Purpose Component Analysis")
print("=" * 60)

# For each component, compute mean absolute effect across all tasks
component_universality = {}
all_components = list(all_results["IOI"]["necessity_mean"].keys())

for comp in all_components:
    nec_effects = [abs(all_results[t]["necessity_mean"].get(comp, 0)) for t in task_names]
    suf_effects = [abs(all_results[t]["sufficiency_mean"].get(comp, 0)) for t in task_names]
    mean_nec = np.mean(nec_effects)
    mean_suf = np.mean(suf_effects)
    std_nec = np.std(nec_effects)
    std_suf = np.std(suf_effects)
    # Coefficient of variation (lower = more universal)
    cv_nec = std_nec / max(mean_nec, 0.001)
    cv_suf = std_suf / max(mean_suf, 0.001)

    component_universality[comp] = {
        "mean_nec": mean_nec,
        "mean_suf": mean_suf,
        "std_nec": std_nec,
        "std_suf": std_suf,
        "cv_nec": cv_nec,
        "cv_suf": cv_suf,
    }

# Rank by mean absolute effect (necessity)
ranked_universal = sorted(component_universality.items(),
                          key=lambda x: x[1]["mean_nec"], reverse=True)

print("\nMost Universal Components (by mean necessity across tasks):")
print(f"{'Component':<15} {'Mean|Nec|':<12} {'Std|Nec|':<12} {'CV':<10} {'Mean|Suf|':<12}")
for comp, info in ranked_universal[:15]:
    label = comp.replace("head_", "H").replace("mlp_", "MLP")
    print(f"  {label:<15} {info['mean_nec']:<12.4f} {info['std_nec']:<12.4f} {info['cv_nec']:<10.2f} {info['mean_suf']:<12.4f}")

# Identify general-purpose components (high effect + low CV)
general_purpose = [comp for comp, info in ranked_universal
                   if info['mean_nec'] > 0.1 and info['cv_nec'] < 0.5]
print(f"\nGeneral-purpose components (mean|nec| > 0.1, CV < 0.5): {len(general_purpose)}")
for comp in general_purpose:
    print(f"  {comp.replace('head_','H').replace('mlp_','M')}")

# ============================================================
# 2. Re-analyze selectivity excluding general-purpose components
# ============================================================
print(f"\n{'='*60}")
print("Selectivity Analysis (Excluding General-Purpose Components)")
print("=" * 60)

TOP_K = 15
selectivity_filtered = []

for target_task in task_names:
    target_results = all_results[target_task]
    other_tasks = [t for t in task_names if t != target_task]

    for int_type_key, int_type_label in [("necessity_mean", "necessity"), ("sufficiency_mean", "sufficiency")]:
        scores = target_results[int_type_key]
        # Exclude general-purpose components before ranking
        filtered_scores = {k: v for k, v in scores.items() if k not in general_purpose}
        ranked = sorted(filtered_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        top_components = [name for name, _ in ranked[:TOP_K]]

        for comp in top_components:
            target_effect = abs(scores[comp])
            other_effects = [abs(all_results[t][int_type_key].get(comp, 0)) for t in other_tasks]
            mean_other = np.mean(other_effects)
            selectivity = target_effect - mean_other

            selectivity_filtered.append({
                "target_task": target_task,
                "component": comp,
                "intervention_type": int_type_label,
                "target_effect": target_effect,
                "mean_other_effect": mean_other,
                "selectivity": selectivity,
                "max_other_effect": max(other_effects),
                "leaks": int(max(other_effects) > 0.05),
            })

sel_filtered_df = pd.DataFrame(selectivity_filtered)

print("\nFiltered Selectivity Summary (excluding general-purpose components):")
print(sel_filtered_df.groupby("intervention_type").agg(
    mean_selectivity=("selectivity", "mean"),
    std_selectivity=("selectivity", "std"),
    mean_leakage=("leaks", "mean"),
    mean_target=("target_effect", "mean"),
    mean_other=("mean_other_effect", "mean"),
).to_string())

# Statistical test
nec_sel = sel_filtered_df[sel_filtered_df["intervention_type"] == "necessity"]["selectivity"]
suf_sel = sel_filtered_df[sel_filtered_df["intervention_type"] == "sufficiency"]["selectivity"]
stat_u, p_u = stats.mannwhitneyu(nec_sel, suf_sel, alternative='two-sided')
d = (suf_sel.mean() - nec_sel.mean()) / np.sqrt((nec_sel.std()**2 + suf_sel.std()**2) / 2)
print(f"\nFiltered Mann-Whitney U: U={stat_u:.1f}, p={p_u:.4f}, Cohen's d={d:.3f}")
print(f"  Necessity: {nec_sel.mean():.4f} ± {nec_sel.std():.4f}")
print(f"  Sufficiency: {suf_sel.mean():.4f} ± {suf_sel.std():.4f}")

# ============================================================
# 3. Bootstrap Confidence Intervals
# ============================================================
print(f"\n{'='*60}")
print("Bootstrap Confidence Intervals for Selectivity Difference")
print("=" * 60)

n_bootstrap = 10000
np.random.seed(42)

nec_vals = nec_sel.values
suf_vals = suf_sel.values

boot_diffs = []
for _ in range(n_bootstrap):
    nec_sample = np.random.choice(nec_vals, size=len(nec_vals), replace=True)
    suf_sample = np.random.choice(suf_vals, size=len(suf_vals), replace=True)
    boot_diffs.append(suf_sample.mean() - nec_sample.mean())

boot_diffs = np.array(boot_diffs)
ci_lower = np.percentile(boot_diffs, 2.5)
ci_upper = np.percentile(boot_diffs, 97.5)
print(f"  Sufficiency - Necessity selectivity difference")
print(f"  Mean: {np.mean(boot_diffs):.4f}")
print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  Contains 0: {'Yes' if ci_lower <= 0 <= ci_upper else 'No'}")

# ============================================================
# 4. Detailed Leakage Analysis
# ============================================================
print(f"\n{'='*60}")
print("Detailed Cross-Task Leakage Patterns")
print("=" * 60)

# For each task, which specific components leak to which other tasks?
for target_task in task_names:
    print(f"\n  --- {target_task} ---")
    for int_type_key, int_label in [("necessity_mean", "Necessity"), ("sufficiency_mean", "Sufficiency")]:
        scores = all_results[target_task][int_type_key]
        ranked = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        other_tasks = [t for t in task_names if t != target_task]

        leaking_comps = []
        for comp, target_eff in ranked:
            other_effs = {t: abs(all_results[t][int_type_key].get(comp, 0)) for t in other_tasks}
            max_other = max(other_effs.values())
            if max_other > 0.05:
                leaking_comps.append((comp, abs(target_eff), other_effs, max_other))

        print(f"  {int_label}: {len(leaking_comps)}/{len(ranked)} top-10 components leak")
        for comp, tgt, others, mx in leaking_comps[:5]:
            label = comp.replace("head_", "H").replace("mlp_", "MLP")
            other_str = ", ".join([f"{t[:3]}={v:.3f}" for t, v in others.items()])
            print(f"    {label}: target={tgt:.3f}, others=[{other_str}]")

# ============================================================
# 5. Visualizations
# ============================================================

# A. Selectivity comparison: all vs filtered
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, df, title in [
    (axes[0], selectivity_df, "All Components"),
    (axes[1], sel_filtered_df, "Excluding General-Purpose"),
]:
    sns.boxplot(data=df, x="intervention_type", y="selectivity", ax=ax)
    sns.stripplot(data=df, x="intervention_type", y="selectivity",
                  color='black', size=3, alpha=0.3, ax=ax)
    ax.set_title(title)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel("Selectivity Score")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/selectivity_filtered_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# B. Component universality plot
fig, ax = plt.subplots(figsize=(10, 6))
comps = [c for c, _ in ranked_universal[:30]]
mean_effects = [component_universality[c]["mean_nec"] for c in comps]
std_effects = [component_universality[c]["std_nec"] for c in comps]
labels = [c.replace("head_", "H").replace("mlp_", "MLP") for c in comps]

colors = ['red' if c in general_purpose else 'steelblue' for c in comps]
ax.barh(range(len(comps)), mean_effects, xerr=std_effects, color=colors, alpha=0.7)
ax.set_yticks(range(len(comps)))
ax.set_yticklabels(labels, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Mean |Necessity Effect| Across Tasks")
ax.set_title("Component Universality (Red = General-Purpose)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/component_universality.png", dpi=150, bbox_inches='tight')
plt.close()

# C. Bootstrap distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(boot_diffs, bins=50, density=True, alpha=0.7, color='steelblue')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
ax.axvline(ci_lower, color='orange', linestyle=':', linewidth=2, label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
ax.axvline(ci_upper, color='orange', linestyle=':', linewidth=2)
ax.axvline(np.mean(boot_diffs), color='green', linewidth=2, label=f'Mean diff: {np.mean(boot_diffs):.3f}')
ax.set_xlabel("Sufficiency Selectivity - Necessity Selectivity")
ax.set_ylabel("Density")
ax.set_title("Bootstrap Distribution of Selectivity Difference\n(Excluding General-Purpose Components)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/bootstrap_selectivity_diff.png", dpi=150, bbox_inches='tight')
plt.close()

# D. Per-component effect profiles
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, task_name in enumerate(task_names):
    ax = axes[idx]
    scores = all_results[task_name]["necessity_mean"]
    ranked = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    other_tasks = [t for t in task_names if t != task_name]

    comp_names = [c.replace("head_", "H").replace("mlp_", "M") for c, _ in ranked]
    target_vals = [abs(v) for _, v in ranked]
    other_vals = [[abs(all_results[t]["necessity_mean"].get(c, 0)) for t in other_tasks] for c, _ in ranked]

    x = np.arange(len(comp_names))
    width = 0.25
    ax.bar(x - width, target_vals, width, label=f'{task_name} (target)', color='steelblue')
    for oi, ot in enumerate(other_tasks):
        ov = [o[oi] for o in other_vals]
        ax.bar(x + width * oi, ov, width, label=f'{ot}', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(comp_names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("|Necessity Effect|")
    ax.set_title(f"Top-15 for {task_name}")
    ax.legend(fontsize=7)
    ax.axhline(0.05, color='red', linestyle='--', alpha=0.5, linewidth=0.5)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/per_component_leakage_profiles.png", dpi=150, bbox_inches='tight')
plt.close()

# Save filtered results
sel_filtered_df.to_csv(f"{RESULTS_DIR}/selectivity_filtered.csv", index=False)

# Save extended summary
extended_summary = {
    "general_purpose_components": general_purpose,
    "n_general_purpose": len(general_purpose),
    "filtered_selectivity": {
        "necessity_mean": float(nec_sel.mean()),
        "necessity_std": float(nec_sel.std()),
        "sufficiency_mean": float(suf_sel.mean()),
        "sufficiency_std": float(suf_sel.std()),
        "mann_whitney_U": float(stat_u),
        "p_value": float(p_u),
        "cohens_d": float(d),
    },
    "bootstrap_ci": {
        "mean_diff": float(np.mean(boot_diffs)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    },
    "original_selectivity": {
        "necessity_mean": float(selectivity_df[selectivity_df["intervention_type"]=="necessity"]["selectivity"].mean()),
        "sufficiency_mean": float(selectivity_df[selectivity_df["intervention_type"]=="sufficiency"]["selectivity"].mean()),
    },
    "leakage_rates_original": summary["leakage_rates"],
    "component_overlap": summary["overlap"],
}

with open(f"{RESULTS_DIR}/extended_summary.json", "w") as f:
    json.dump(extended_summary, f, indent=2)

print(f"\n{'='*60}")
print("Extended analysis complete. Files saved.")
print("=" * 60)
