# Cloned Repositories

## Repo 1: TransformerLens
- **URL**: https://github.com/TransformerLensOrg/TransformerLens
- **Purpose**: Core library for mechanistic interpretability of GPT-style LMs. Provides HookPoints on every activation for surgical interventions (ablation, activation patching).
- **Location**: code/TransformerLens/
- **Key files**: `transformer_lens/` package, demos in `demos/`
- **How to use**: `pip install transformer_lens`. Supports 50+ open-source LMs. Central tool for running necessity (ablation) and sufficiency (patching) experiments.

## Repo 2: pyvene
- **URL**: https://github.com/stanfordnlp/pyvene
- **Purpose**: Stanford NLP's library for causal interventions on any PyTorch model. Implements DAS, interchange interventions, causal abstraction analysis.
- **Location**: code/pyvene/
- **Key files**: `pyvene/` package, tutorials in `tutorials/`
- **How to use**: `pip install pyvene`. Directly implements the causal abstraction framework that formalizes necessity and sufficiency. Used by MIB benchmark.

## Repo 3: CausalGym
- **URL**: https://github.com/aryamanarora/causalgym
- **Purpose**: Benchmark for comparing causal interpretability methods on 29 linguistic tasks. Introduces selectivity metric via control tasks.
- **Location**: code/causalgym/
- **Key files**: `causalgym/` package, `scripts/` for running experiments
- **How to use**: Contains evaluation code for DAS, probing, difference-in-means, PCA, k-means on Pythia models. Selectivity = real task performance - control task performance.

## Repo 4: Feature Circuits
- **URL**: https://github.com/saprmarks/feature-circuits
- **Purpose**: SAE-based circuit discovery. Finds sparse feature circuits using attribution patching over SAE features. Includes SHIFT method for removing task-irrelevant features.
- **Location**: code/feature-circuits/
- **Key files**: Core circuit discovery code, pre-trained SAEs
- **How to use**: Discovers circuits with ~100 features vs ~1500 neurons. Evaluates faithfulness (sufficiency) and completeness (necessity).

## Repo 5: MIB (Mechanistic Interpretability Benchmark)
- **URL**: https://github.com/aaronmueller/MIB
- **Purpose**: Two-track benchmark: circuit localization (EAP, EAP-IG, UGS) and causal variable localization (DAS, DBM, SAE features). Published at ICML 2025.
- **Location**: code/MIB/
- **Key files**: Benchmark evaluation code, submission format examples
- **How to use**: Provides CPR (sufficiency) and CMD (necessity) metrics for circuits; IIA for causal variables. Supports GPT-2 Small, Qwen-2.5, Gemma-2, Llama-3.1.
