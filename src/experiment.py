"""
Necessity, Sufficiency, and Selectivity in Mechanistic Interpretability

This experiment measures:
1. Necessity scores (via noising/ablation) for attention heads and MLPs across 3 tasks
2. Sufficiency scores (via denoising/activation patching) for the same components
3. Cross-task selectivity: does a component identified for task A also affect tasks B, C?

Model: GPT-2 Small (12 layers, 12 heads per layer = 144 heads + 12 MLPs = 156 components)
Tasks: IOI (indirect object identification), Subject-Verb Agreement, Greater-Than
"""

import os
import sys
import json
import random
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# TransformerLens
import transformer_lens
from transformer_lens import HookedTransformer, utils

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_LAYERS = 12
N_HEADS = 12
N_EXAMPLES = 100  # per task for patching experiments

RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"TransformerLens: {getattr(transformer_lens, '__version__', 'installed')}")


# ============================================================
# Task Definitions
# ============================================================

def make_ioi_dataset(n=N_EXAMPLES):
    """Create IOI task examples with clean/corrupted pairs.

    IOI: "When Mary and John went to the store, John gave a drink to" -> Mary
    Corrupted: replace names so the answer changes.
    """
    templates = [
        "When {A} and {B} went to the {place}, {B2} gave a {obj} to",
        "Then, {A} and {B} had a lot of fun at the {place}. {B2} gave a {obj} to",
        "After {A} and {B} arrived at the {place}, {B2} handed a {obj} to",
    ]
    # Only use names that are single tokens with leading space in GPT-2
    names = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry",
             "Jack", "Kate", "Mary", "Nick", "Peter", "Sam", "Tom",
             "John", "Sarah", "James", "Lisa", "Mark", "Anna", "Paul", "Amy",
             "Luke", "Dean", "Dan"]
    places = ["store", "park", "school", "office", "restaurant", "library", "museum"]
    objects = ["drink", "book", "letter", "gift", "flower", "ball", "toy"]

    examples = []
    for i in range(n):
        template = templates[i % len(templates)]
        a, b = random.sample(names, 2)
        place = random.choice(places)
        obj = random.choice(objects)

        # Clean: subject=B repeats, answer=A (indirect object)
        clean = template.format(A=a, B=b, B2=b, place=place, obj=obj)

        # Corrupted: use different names -> changes answer
        c, d = random.sample([nm for nm in names if nm not in [a, b]], 2)
        corrupted = template.format(A=c, B=d, B2=d, place=place, obj=obj)

        examples.append({
            "clean": clean,
            "corrupted": corrupted,
            "answer_token": " " + a,   # with space prefix for GPT-2 tokenization
            "wrong_token": " " + b,
            "task": "ioi"
        })
    return examples


def make_sv_agreement_dataset(n=N_EXAMPLES):
    """Create subject-verb agreement examples.

    "The cat on the mats sleeps" (good) vs "The cat on the mats sleep" (bad)
    We measure logit difference between correct and incorrect verb forms.
    """
    # Singular subjects with plural attractors
    subjects_sg = ["cat", "dog", "boy", "girl", "man", "teacher", "doctor", "bird"]
    attractors_pl = ["mats", "tables", "chairs", "books", "cars", "boxes", "trees", "walls"]
    preps = ["near the", "beside the", "behind the", "on the", "with the", "above the"]

    # Plural subjects with singular attractors
    subjects_pl = ["cats", "dogs", "boys", "girls", "men", "teachers", "doctors", "birds"]
    attractors_sg = ["mat", "table", "chair", "book", "car", "box", "tree", "wall"]

    examples = []
    for i in range(n):
        if i % 2 == 0:
            # Singular subject
            subj = random.choice(subjects_sg)
            attr = random.choice(attractors_pl)
            prep = random.choice(preps)
            clean = f"The {subj} {prep} {attr}"
            corrupted = f"The {random.choice(subjects_pl)} {prep} {random.choice(attractors_sg)}"
            answer_token = " is"      # singular verb
            wrong_token = " are"      # plural verb
        else:
            # Plural subject
            subj = random.choice(subjects_pl)
            attr = random.choice(attractors_sg)
            prep = random.choice(preps)
            clean = f"The {subj} {prep} {attr}"
            corrupted = f"The {random.choice(subjects_sg)} {prep} {random.choice(attractors_pl)}"
            answer_token = " are"     # plural verb
            wrong_token = " is"       # singular verb

        examples.append({
            "clean": clean,
            "corrupted": corrupted,
            "answer_token": answer_token,
            "wrong_token": wrong_token,
            "task": "sv_agreement"
        })
    return examples


def make_greater_than_dataset(n=N_EXAMPLES):
    """Create greater-than task examples.

    "The war lasted from 1732 to 17" -> next token should be >= 32
    Corrupted: change the start year so the constraint changes.
    """
    examples = []
    for i in range(n):
        # Pick a start year in the 1700s
        start_decade = random.randint(3, 8)  # 30-89
        start_unit = random.randint(0, 9)
        start = start_decade * 10 + start_unit

        # The answer should be a digit >= start_decade (for the decade digit)
        clean = f"The war lasted from 17{start:02d} to 17"

        # Corrupted: different start year
        corrupt_decade = random.randint(3, 8)
        while corrupt_decade == start_decade:
            corrupt_decade = random.randint(3, 8)
        corrupt_unit = random.randint(0, 9)
        corrupt_start = corrupt_decade * 10 + corrupt_unit
        corrupted = f"The war lasted from 17{corrupt_start:02d} to 17"

        # For logit diff: we compare logits for valid decades (>= start_decade) vs invalid (< start_decade)
        # We use the decade digits as answer/wrong tokens
        answer_token = str(min(start_decade + 1, 9))  # a valid continuation
        wrong_token = str(max(start_decade - 2, 0))    # an invalid continuation

        examples.append({
            "clean": clean,
            "corrupted": corrupted,
            "answer_token": answer_token,
            "wrong_token": wrong_token,
            "task": "greater_than"
        })
    return examples


# ============================================================
# Patching Infrastructure
# ============================================================

def get_logit_diff(logits, answer_ids, wrong_ids):
    """Compute logit difference: logit(answer) - logit(wrong) at last position."""
    last_logits = logits[:, -1, :]  # [batch, vocab]
    answer_logits = last_logits.gather(1, answer_ids.unsqueeze(1)).squeeze(1)
    wrong_logits = last_logits.gather(1, wrong_ids.unsqueeze(1)).squeeze(1)
    return (answer_logits - wrong_logits).mean().item()


def prepare_batch(model, examples):
    """Tokenize examples and get answer/wrong token ids."""
    # Use padding to handle variable lengths
    model.tokenizer.pad_token = model.tokenizer.eos_token

    clean_texts = [e["clean"] for e in examples]
    corrupted_texts = [e["corrupted"] for e in examples]

    clean_tokens = model.to_tokens(clean_texts, prepend_bos=True)
    corrupted_tokens = model.to_tokens(corrupted_texts, prepend_bos=True)

    # Pad to same length
    max_len = max(clean_tokens.shape[1], corrupted_tokens.shape[1])
    if clean_tokens.shape[1] < max_len:
        clean_tokens = F.pad(clean_tokens, (0, max_len - clean_tokens.shape[1]), value=model.tokenizer.eos_token_id)
    if corrupted_tokens.shape[1] < max_len:
        corrupted_tokens = F.pad(corrupted_tokens, (0, max_len - corrupted_tokens.shape[1]), value=model.tokenizer.eos_token_id)

    answer_ids = torch.tensor([model.to_single_token(e["answer_token"]) for e in examples], device=DEVICE)
    wrong_ids = torch.tensor([model.to_single_token(e["wrong_token"]) for e in examples], device=DEVICE)

    return clean_tokens, corrupted_tokens, answer_ids, wrong_ids


def run_patching_experiment(model, examples, batch_size=25):
    """Run both noising (necessity) and denoising (sufficiency) patching.

    Noising (necessity): run on clean input, patch in corrupted activations.
        If logit diff drops → component is necessary.

    Denoising (sufficiency): run on corrupted input, patch in clean activations.
        If logit diff recovers → component is sufficient.

    Returns:
        necessity_scores: dict mapping component_name -> mean effect
        sufficiency_scores: dict mapping component_name -> mean effect
    """
    necessity_scores = defaultdict(list)
    sufficiency_scores = defaultdict(list)

    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        clean_tokens, corrupted_tokens, answer_ids, wrong_ids = prepare_batch(model, batch)

        # Get clean and corrupted caches
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens)
            _, corrupted_cache = model.run_with_cache(corrupted_tokens)

            clean_logits = model(clean_tokens)
            corrupted_logits = model(corrupted_tokens)

        clean_ld = get_logit_diff(clean_logits, answer_ids, wrong_ids)
        corrupted_ld = get_logit_diff(corrupted_logits, answer_ids, wrong_ids)

        total_effect = clean_ld - corrupted_ld
        if abs(total_effect) < 0.01:
            continue  # skip if no signal

        # Patch each attention head and MLP layer
        for layer in range(N_LAYERS):
            # --- Attention heads ---
            for head in range(N_HEADS):
                hook_name = utils.get_act_name("z", layer)

                # Noising (necessity): clean run, patch in corrupted head output
                def noising_hook(activation, hook, l=layer, h=head):
                    activation[:, :, h, :] = corrupted_cache[hook.name][:, :activation.shape[1], h, :]
                    return activation

                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        clean_tokens,
                        fwd_hooks=[(hook_name, noising_hook)]
                    )
                patched_ld = get_logit_diff(patched_logits, answer_ids, wrong_ids)
                # Necessity = how much logit diff drops (normalized)
                nec = (clean_ld - patched_ld) / max(abs(total_effect), 0.01)
                necessity_scores[f"head_{layer}_{head}"].append(nec)

                # Denoising (sufficiency): corrupted run, patch in clean head output
                def denoising_hook(activation, hook, l=layer, h=head):
                    activation[:, :, h, :] = clean_cache[hook.name][:, :activation.shape[1], h, :]
                    return activation

                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        corrupted_tokens,
                        fwd_hooks=[(hook_name, denoising_hook)]
                    )
                patched_ld = get_logit_diff(patched_logits, answer_ids, wrong_ids)
                # Sufficiency = how much logit diff recovers (normalized)
                suf = (patched_ld - corrupted_ld) / max(abs(total_effect), 0.01)
                sufficiency_scores[f"head_{layer}_{head}"].append(suf)

            # --- MLP layers ---
            mlp_hook_name = utils.get_act_name("mlp_out", layer)

            # Noising MLP
            def noising_mlp_hook(activation, hook, l=layer):
                activation[:, :, :] = corrupted_cache[hook.name][:, :activation.shape[1], :]
                return activation

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    clean_tokens,
                    fwd_hooks=[(mlp_hook_name, noising_mlp_hook)]
                )
            patched_ld = get_logit_diff(patched_logits, answer_ids, wrong_ids)
            nec = (clean_ld - patched_ld) / max(abs(total_effect), 0.01)
            necessity_scores[f"mlp_{layer}"].append(nec)

            # Denoising MLP
            def denoising_mlp_hook(activation, hook, l=layer):
                activation[:, :, :] = clean_cache[hook.name][:, :activation.shape[1], :]
                return activation

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(mlp_hook_name, denoising_mlp_hook)]
                )
            patched_ld = get_logit_diff(patched_logits, answer_ids, wrong_ids)
            suf = (patched_ld - corrupted_ld) / max(abs(total_effect), 0.01)
            sufficiency_scores[f"mlp_{layer}"].append(suf)

        # Free memory
        del clean_cache, corrupted_cache
        torch.cuda.empty_cache()

    # Average across batches
    necessity_mean = {k: np.mean(v) for k, v in necessity_scores.items()}
    sufficiency_mean = {k: np.mean(v) for k, v in sufficiency_scores.items()}
    necessity_std = {k: np.std(v) for k, v in necessity_scores.items()}
    sufficiency_std = {k: np.std(v) for k, v in sufficiency_scores.items()}

    return {
        "necessity_mean": necessity_mean,
        "sufficiency_mean": sufficiency_mean,
        "necessity_std": necessity_std,
        "sufficiency_std": sufficiency_std,
        "clean_logit_diff": clean_ld,
        "corrupted_logit_diff": corrupted_ld,
    }


# ============================================================
# Analysis Functions
# ============================================================

def compute_selectivity(all_results, task_names, top_k=15):
    """Compute cross-task selectivity for top-K components.

    For each task, identify top-K components by necessity and sufficiency.
    Then measure those components' effects on all other tasks.

    Selectivity = effect on target task - mean effect on non-target tasks.
    """
    selectivity_data = []

    for target_task in task_names:
        target_results = all_results[target_task]
        other_tasks = [t for t in task_names if t != target_task]

        for intervention_type in ["necessity_mean", "sufficiency_mean"]:
            scores = target_results[intervention_type]
            # Sort by absolute effect
            ranked = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
            top_components = [name for name, _ in ranked[:top_k]]

            for comp in top_components:
                target_effect = abs(scores[comp])
                other_effects = [abs(all_results[t][intervention_type].get(comp, 0)) for t in other_tasks]
                mean_other = np.mean(other_effects)
                selectivity = target_effect - mean_other

                int_type = "necessity" if "necessity" in intervention_type else "sufficiency"
                selectivity_data.append({
                    "target_task": target_task,
                    "component": comp,
                    "intervention_type": int_type,
                    "target_effect": target_effect,
                    "mean_other_effect": mean_other,
                    "selectivity": selectivity,
                    "max_other_effect": max(other_effects) if other_effects else 0,
                    "leaks": int(max(other_effects) > 0.05 if other_effects else False),
                })

    return pd.DataFrame(selectivity_data)


def compute_rank_correlation(results, task_name):
    """Compute Spearman correlation between necessity and sufficiency rankings."""
    nec = results[task_name]["necessity_mean"]
    suf = results[task_name]["sufficiency_mean"]

    components = sorted(nec.keys())
    nec_vals = [abs(nec[c]) for c in components]
    suf_vals = [abs(suf[c]) for c in components]

    rho, pval = stats.spearmanr(nec_vals, suf_vals)
    return rho, pval, components, nec_vals, suf_vals


# ============================================================
# Visualization
# ============================================================

def plot_patching_heatmaps(results, task_name, save_prefix):
    """Plot necessity and sufficiency heatmaps for attention heads."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (metric, title) in enumerate([
        ("necessity_mean", f"Necessity (Noising) - {task_name}"),
        ("sufficiency_mean", f"Sufficiency (Denoising) - {task_name}")
    ]):
        scores = results[task_name][metric]
        matrix = np.zeros((N_LAYERS, N_HEADS))
        for layer in range(N_LAYERS):
            for head in range(N_HEADS):
                matrix[layer, head] = scores.get(f"head_{layer}_{head}", 0)

        im = axes[idx].imshow(matrix, aspect='auto', cmap='RdBu_r',
                               vmin=-0.3, vmax=0.3)
        axes[idx].set_xlabel("Head")
        axes[idx].set_ylabel("Layer")
        axes[idx].set_title(title)
        plt.colorbar(im, ax=axes[idx], label="Normalized Effect")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{save_prefix}_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_necessity_vs_sufficiency(results, task_name, save_prefix):
    """Scatter plot of necessity vs sufficiency scores."""
    rho, pval, components, nec_vals, suf_vals = compute_rank_correlation(results, task_name)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color by component type
    colors = ['#e74c3c' if 'mlp' in c else '#3498db' for c in components]
    sizes = [80 if 'mlp' in c else 30 for c in components]

    ax.scatter(nec_vals, suf_vals, c=colors, s=sizes, alpha=0.6)

    # Label top components
    combined = [(n + s, c, n, s) for c, n, s in zip(components, nec_vals, suf_vals)]
    combined.sort(reverse=True)
    for _, comp, nec, suf in combined[:10]:
        ax.annotate(comp.replace("head_", "H").replace("mlp_", "MLP"),
                    (nec, suf), fontsize=7, alpha=0.8)

    max_val = max(max(nec_vals), max(suf_vals)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel("Necessity Score (|effect of ablation|)")
    ax.set_ylabel("Sufficiency Score (|effect of patching|)")
    ax.set_title(f"Necessity vs Sufficiency - {task_name}\nSpearman ρ={rho:.3f}, p={pval:.1e}")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#3498db', label='Attention Head'),
        Patch(color='#e74c3c', label='MLP Layer'),
    ], loc='upper left')

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{save_prefix}_nec_vs_suf.png", dpi=150, bbox_inches='tight')
    plt.close()

    return rho, pval


def plot_selectivity_comparison(selectivity_df, save_prefix):
    """Compare selectivity between necessity and sufficiency-identified components."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Selectivity by intervention type (boxplot)
    ax = axes[0]
    sns.boxplot(data=selectivity_df, x="intervention_type", y="selectivity", ax=ax)
    ax.set_title("Selectivity by Intervention Type")
    ax.set_ylabel("Selectivity (target - mean other)")
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # 2. Selectivity by task (boxplot)
    ax = axes[1]
    sns.boxplot(data=selectivity_df, x="target_task", y="selectivity",
                hue="intervention_type", ax=ax)
    ax.set_title("Selectivity by Task and Intervention")
    ax.set_ylabel("Selectivity")
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

    # 3. Leakage rate
    ax = axes[2]
    leakage = selectivity_df.groupby(["target_task", "intervention_type"])["leaks"].mean()
    leakage_df = leakage.reset_index()
    sns.barplot(data=leakage_df, x="target_task", y="leaks",
                hue="intervention_type", ax=ax)
    ax.set_title("Cross-Task Leakage Rate")
    ax.set_ylabel("Fraction of Top-K with Leakage")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{save_prefix}_selectivity.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_cross_task_heatmap(all_results, task_names, top_k=10, save_prefix="cross_task"):
    """Heatmap showing top components from each task and their effects on all tasks."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for idx, intervention_type in enumerate(["necessity_mean", "sufficiency_mean"]):
        title = "Necessity" if "necessity" in intervention_type else "Sufficiency"

        # Collect top components across tasks
        all_comps = []
        comp_labels = []
        for task in task_names:
            scores = all_results[task][intervention_type]
            ranked = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
            for name, _ in ranked[:top_k]:
                if name not in all_comps:
                    all_comps.append(name)
                    comp_labels.append(f"{name}")

        # Build effect matrix
        matrix = np.zeros((len(all_comps), len(task_names)))
        for j, task in enumerate(task_names):
            for i, comp in enumerate(all_comps):
                matrix[i, j] = abs(all_results[task][intervention_type].get(comp, 0))

        ax = axes[idx]
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(task_names)))
        ax.set_xticklabels(task_names, rotation=45, ha='right')
        ax.set_yticks(range(len(comp_labels)))
        ax.set_yticklabels([c.replace("head_", "H").replace("mlp_", "M") for c in comp_labels], fontsize=7)
        ax.set_title(f"Cross-Task Component Effects ({title})")
        plt.colorbar(im, ax=ax, label="|Normalized Effect|")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{save_prefix}_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Main Execution
# ============================================================

def main():
    print("=" * 60)
    print("Loading GPT-2 Small...")
    print("=" * 60)

    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()
    print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")

    # Create datasets
    print("\nCreating task datasets...")
    tasks = {
        "IOI": make_ioi_dataset(N_EXAMPLES),
        "SV_Agreement": make_sv_agreement_dataset(N_EXAMPLES),
        "Greater_Than": make_greater_than_dataset(N_EXAMPLES),
    }

    for name, data in tasks.items():
        print(f"  {name}: {len(data)} examples")
        print(f"    Sample: '{data[0]['clean']}' -> {data[0]['answer_token']}")

    # Run patching experiments for each task
    all_results = {}
    for task_name, examples in tasks.items():
        print(f"\n{'='*60}")
        print(f"Running patching for: {task_name}")
        print(f"{'='*60}")
        t0 = time.time()
        results = run_patching_experiment(model, examples, batch_size=25)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")
        print(f"  Clean logit diff: {results['clean_logit_diff']:.3f}")
        print(f"  Corrupted logit diff: {results['corrupted_logit_diff']:.3f}")

        # Top components
        for metric_name, metric_key in [("Necessity", "necessity_mean"), ("Sufficiency", "sufficiency_mean")]:
            scores = results[metric_key]
            ranked = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            print(f"  Top-5 {metric_name}:")
            for comp, score in ranked:
                print(f"    {comp}: {score:.4f}")

        all_results[task_name] = results

    task_names = list(tasks.keys())

    # ---- Analysis ----
    print(f"\n{'='*60}")
    print("Analysis")
    print(f"{'='*60}")

    # 1. Rank correlations (necessity vs sufficiency)
    rank_corrs = {}
    for task_name in task_names:
        rho, pval = plot_necessity_vs_sufficiency(all_results, task_name, task_name.lower())
        rank_corrs[task_name] = {"rho": rho, "pval": pval}
        print(f"\n  {task_name}: Necessity-Sufficiency Spearman ρ = {rho:.3f} (p = {pval:.2e})")

    # 2. Cross-task selectivity
    selectivity_df = compute_selectivity(all_results, task_names, top_k=15)
    plot_selectivity_comparison(selectivity_df, "overall")

    # Summary statistics
    print("\n  Selectivity Summary:")
    sel_summary = selectivity_df.groupby("intervention_type").agg(
        mean_selectivity=("selectivity", "mean"),
        std_selectivity=("selectivity", "std"),
        mean_leakage=("leaks", "mean"),
        mean_target_effect=("target_effect", "mean"),
        mean_other_effect=("mean_other_effect", "mean"),
    )
    print(sel_summary.to_string())

    # Statistical test: selectivity difference between necessity and sufficiency
    nec_sel = selectivity_df[selectivity_df["intervention_type"] == "necessity"]["selectivity"]
    suf_sel = selectivity_df[selectivity_df["intervention_type"] == "sufficiency"]["selectivity"]

    # Use Mann-Whitney U test (non-parametric)
    stat_u, p_u = stats.mannwhitneyu(nec_sel, suf_sel, alternative='two-sided')
    effect_size_d = (suf_sel.mean() - nec_sel.mean()) / np.sqrt((nec_sel.std()**2 + suf_sel.std()**2) / 2)
    print(f"\n  Mann-Whitney U test (selectivity): U={stat_u:.1f}, p={p_u:.4f}")
    print(f"  Effect size (Cohen's d): {effect_size_d:.3f}")
    print(f"  Mean selectivity - Necessity: {nec_sel.mean():.4f} ± {nec_sel.std():.4f}")
    print(f"  Mean selectivity - Sufficiency: {suf_sel.mean():.4f} ± {suf_sel.std():.4f}")

    # 3. Cross-task overlap analysis
    print("\n  Cross-Task Component Overlap (Top-15):")
    overlap_data = {}
    for int_type in ["necessity_mean", "sufficiency_mean"]:
        int_label = "necessity" if "necessity" in int_type else "sufficiency"
        top_sets = {}
        for task_name in task_names:
            scores = all_results[task_name][int_type]
            ranked = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
            top_sets[task_name] = set([name for name, _ in ranked[:15]])

        for i, t1 in enumerate(task_names):
            for t2 in task_names[i+1:]:
                overlap = len(top_sets[t1] & top_sets[t2])
                key = f"{t1} ∩ {t2}"
                if key not in overlap_data:
                    overlap_data[key] = {}
                overlap_data[key][int_label] = overlap
                print(f"    {int_label}: {key} = {overlap}/15 ({overlap/15*100:.0f}%)")

    # 4. Heatmaps
    for task_name in task_names:
        plot_patching_heatmaps(all_results, task_name, task_name.lower())

    plot_cross_task_heatmap(all_results, task_names, top_k=10)

    # Per-task selectivity test
    print("\n  Per-Task Selectivity Tests:")
    for task_name in task_names:
        task_data = selectivity_df[selectivity_df["target_task"] == task_name]
        nec_s = task_data[task_data["intervention_type"] == "necessity"]["selectivity"]
        suf_s = task_data[task_data["intervention_type"] == "sufficiency"]["selectivity"]
        if len(nec_s) > 1 and len(suf_s) > 1:
            t_stat, t_pval = stats.mannwhitneyu(nec_s, suf_s, alternative='two-sided')
            print(f"    {task_name}: nec_sel={nec_s.mean():.4f}, suf_sel={suf_s.mean():.4f}, p={t_pval:.4f}")

    # ---- Save Results ----
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")

    # Convert results to JSON-serializable format
    results_json = {}
    for task_name in task_names:
        r = all_results[task_name]
        results_json[task_name] = {
            "necessity_mean": r["necessity_mean"],
            "sufficiency_mean": r["sufficiency_mean"],
            "necessity_std": r["necessity_std"],
            "sufficiency_std": r["sufficiency_std"],
            "clean_logit_diff": r["clean_logit_diff"],
            "corrupted_logit_diff": r["corrupted_logit_diff"],
        }

    with open(f"{RESULTS_DIR}/patching_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    selectivity_df.to_csv(f"{RESULTS_DIR}/selectivity_data.csv", index=False)

    # Save summary
    summary = {
        "config": {
            "model": "gpt2",
            "n_examples": N_EXAMPLES,
            "seed": SEED,
            "device": DEVICE,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "top_k": 15,
        },
        "rank_correlations": rank_corrs,
        "selectivity_test": {
            "statistic": float(stat_u),
            "p_value": float(p_u),
            "effect_size_d": float(effect_size_d),
            "necessity_mean_selectivity": float(nec_sel.mean()),
            "sufficiency_mean_selectivity": float(suf_sel.mean()),
        },
        "overlap": overlap_data,
        "per_task_logit_diffs": {
            t: {
                "clean": all_results[t]["clean_logit_diff"],
                "corrupted": all_results[t]["corrupted_logit_diff"],
            } for t in task_names
        },
        "leakage_rates": {
            "necessity": float(selectivity_df[selectivity_df["intervention_type"]=="necessity"]["leaks"].mean()),
            "sufficiency": float(selectivity_df[selectivity_df["intervention_type"]=="sufficiency"]["leaks"].mean()),
        }
    }

    with open(f"{RESULTS_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Plots saved to {PLOTS_DIR}/")
    print("Done!")

    return all_results, selectivity_df, summary


if __name__ == "__main__":
    all_results, selectivity_df, summary = main()
