# Downloaded Datasets

This directory contains datasets for the research project on necessity, sufficiency, and selectivity in mechanistic interpretability. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: MIB IOI (Indirect Object Identification)

### Overview
- **Source**: HuggingFace `mib-bench/ioi`
- **Size**: 10,000 train / 10,000 validation / 1,000 test
- **Format**: HuggingFace Dataset
- **Task**: Predict the indirect object in sentences like "When Mary and John went to the store, John gave an apple to ___"
- **License**: Research use

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("mib-bench/ioi")
dataset.save_to_disk("datasets/mib_ioi")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/mib_ioi")
```

### Features
- `prompt`: Input sentence
- `choices`: Answer options
- `answerKey`: Correct answer
- 8 counterfactual types: `abc_counterfactual`, `random_names_counterfactual`, `s1_io_flip_counterfactual`, etc.

### Notes
- Each instance includes 8 types of counterfactual pairs for causal intervention experiments
- Canonical task for mechanistic interpretability research
- Used in MIB benchmark for both circuit localization and causal variable localization

---

## Dataset 2: MIB Arithmetic Addition

### Overview
- **Source**: HuggingFace `mib-bench/arithmetic_addition`
- **Size**: 34,440 train / 4,920 validation / 1,000 test
- **Format**: HuggingFace Dataset
- **Task**: Two-digit addition (e.g., "What is the sum of 13 and 25?")
- **License**: Research use

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("mib-bench/arithmetic_addition")
dataset.save_to_disk("datasets/mib_arithmetic_addition")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/mib_arithmetic_addition")
```

### Features
- `prompt`: Input question
- `label`: Correct answer
- `operand1`, `operand2`: The two operands
- 7 counterfactual types: `random_counterfactual`, `ones_op1_counterfactual`, `tens_op1_counterfactual`, `ones_carry_counterfactual`, etc.

### Notes
- 6 natural language templates per operand pair
- Counterfactuals vary individual digits and carry values
- MIB found carry-the-one variable is difficult to localize in linear feature space

---

## Dataset 3: MIB MCQA (Multiple-Choice QA)

### Overview
- **Source**: HuggingFace `mib-bench/copycolors_mcqa` (config: `4_answer_choices`)
- **Size**: 110 train / 50 validation / 50 test
- **Format**: HuggingFace Dataset
- **Task**: Synthetic object color QA (e.g., "A box is brown. What color is a box? A. gray B. black C. white D. brown")
- **License**: Research use

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("mib-bench/copycolors_mcqa", "4_answer_choices")
dataset.save_to_disk("datasets/mib_mcqa")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/mib_mcqa")
```

### Features
- `prompt`: Input question with choices
- `answerKey`: Correct answer letter
- 9 counterfactual types: `noun_color_counterfactual`, `noun_counterfactual`, `color_counterfactual`, `answerPosition_counterfactual`, `symbol_counterfactual`, etc.

### Notes
- Small dataset (low-data regime) -- tests method robustness with limited examples
- Separates semantic counterfactuals (noun, color) from format counterfactuals (position, symbol)
- DAS achieves up to 100% IIA at best layers

---

## Dataset 4: BLiMP Subsets (Linguistic Minimal Pairs)

### Overview
- **Source**: HuggingFace `nyu-mll/blimp`
- **Configs downloaded**: `anaphor_number_agreement`, `regular_plural_subject_verb_agreement_1`, `distractor_agreement_relative_clause`
- **Size**: 1,000 minimal pairs per config
- **Format**: HuggingFace Dataset
- **Task**: Evaluate grammatical knowledge via minimal pairs

### Download Instructions

```python
from datasets import load_dataset
for config in ["anaphor_number_agreement", "regular_plural_subject_verb_agreement_1", "distractor_agreement_relative_clause"]:
    ds = load_dataset("nyu-mll/blimp", config)
    ds.save_to_disk(f"datasets/blimp_{config}")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/blimp_anaphor_number_agreement")
```

### Notes
- 67 total BLiMP configs available; 3 agreement-related ones downloaded
- Minimal pairs enable clean causal testing of discovered features
- Used for cross-task selectivity evaluation: test whether features found for one agreement task also affect others

---

## Sample Data

Sample data files are included in this directory:
- `mib_ioi_samples.json` - 3 example IOI instances with counterfactuals
- `mib_arithmetic_samples.json` - 3 example arithmetic instances with counterfactuals
