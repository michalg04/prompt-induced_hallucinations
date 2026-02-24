# Mechanisms of Prompt-Induced Hallucination in Vision–Language Models
<img width="922" height="428" alt="Screenshot 2026-02-24 at 4 13 27 PM" src="https://github.com/user-attachments/assets/1d87c718-e2c8-4380-92ab-76e3ba7dd738" />

This repository contains code for the paper [Mechanisms of Prompt-Induced Hallucination in Vision–Language Models](https://arxiv.org/pdf/2601.05201). The repository includes:

- Baseline evaluation (counting + color tasks)
- Attention head knockouts
- Identification of influential heads
- Attention mass analysis
- Token probability analysis under intervention

Supported models:

- `qwen` — Qwen2-VL-7B-Instruct  
- `llava-one` — LLaVA-OneVision (Qwen2-7B backbone)  
- `janus` — Janus-Pro-7B  

---

# Requirements

```bash
pip install -r requirements.txt
```

---
# Tutorial

Quick example from counting dataset to show how to use our PIH ablation to remove copying:

```
tutorial_pih.ipynb
```

---
# Full pipeline
# Step 1 — Create Counting Dataset

Open:

```
1_create_counting_dataset.ipynb
```

Run all cells to generate:

```
counting_with_prompts.csv
```

This file is required for the counting task.

---

# Step 2 — Run Baseline Evaluation

Script:

```
2_evaluate_vlms.py
```

## Color Task

```bash
python 2_evaluate_vlms.py \
  --model_version qwen \
  --dataset_size full \
  --task color
```

## Counting Task

```bash
python 2_evaluate_vlms.py \
  --model_version qwen \
  --dataset_size full \
  --task count
```

### Output

```
{model}_{task}_results_{dataset_size}.csv
```

Example:

```
qwen_color_results_full.csv
```

---

# Step 3 — Run Head Knockout Sweep

Script:

```
3_knockouts.py
```

This sweeps attention heads and records response changes.

## Full Sweep (discover influential heads)

```bash
python 3_knockouts.py \
  --model_version qwen \
  --task count \
  --dataset_size full \
  --evaluation False \
  --layer_lower 0 \
  --layer_upper 32 \
  --head_lower 0 \
  --head_upper 32
```

Output:

```
all_PIH_ablation_results_L0H0_to_L32H32_qwen_count_full.csv
```

---

## Evaluate Best Heads Only

```bash
python 3_knockouts.py \
  --model_version qwen \
  --task count \
  --dataset_size full \
  --evaluation True \
  --heads_to_ablate best3
```

Output:

```
best_PIH_ablation_results_best3_qwen_count_full.csv
```

---

# Step 4 — Evaluate Knockout Results

Script:

```
4_evaluate_knockouts.py
```

This computes:

- Correction rates
- Head rankings
- Aggregate statistics

Example:

```bash
python 4_evaluate_knockouts.py \
  --model_version qwen \
  --task count \
  --dataset_size full \
  --eval_type all \
  --layer_lower 0 \
  --layer_upper 32 \
  --head_lower 0 \
  --head_upper 32
```

---

# Step 5 — Attention Mass Analysis

Script:

```
5_attention_mass.py
```

Computes how much attention mass flows to image tokens before vs after ablation.

Example:

```bash
python 5_attention_mass.py \
  --model_version qwen \
  --task count \
  --dataset_size full \
  --evaluation True \
  --heads_to_ablate best3
```

Output:

```
attn_mass_per_layer_qwen_best3.pkl
```

---

# Step 6 — Extract Full Attention Maps

Script:

```
6_attention_outputs.py
```

Saves full attention matrices for selected heads.

Example:

```bash
python 6_attention_outputs.py \
  --model_version qwen \
  --task count \
  --dataset_size full \
  --heads_to_ablate best3 \
  --num_samples 20
```

Output:

```
attention_results_qwen_count_best3_baseline_20.pkl
```

---

# Step 7 — Token Probability Analysis

Script:

```
6_generate_with_probabilities_knockout.py
```

Computes token probabilities for:

- Ground truth answers
- Corrupt answers
- Under head ablation

Example:

```bash
python 6_generate_with_probabilities_knockout.py \
  --model_version qwen \
  --task count \
  --heads_to_ablate best3 \
  --num_samples 50 \
  --alpha 0.0
```

Output:

```
qwen_count_confidence_by_class_knockout_0.0_decoupled.pkl
```

---


# Citation

If you use this code, please cite:

> Mechanisms of Prompt-Induced Hallucination in Vision–Language Models

