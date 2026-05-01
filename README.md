# moe-caption-specialization

Analyzing whether MoE (Mixture-of-Experts) routers learn linguistically meaningful expert specialization when encoding natural language text.

## Research Question

When a Switch Transformer encodes text, do its experts naturally specialize along linguistic dimensions (e.g., nouns → expert A, verbs → expert B)? Or is routing driven primarily by load balancing rather than semantic structure?

We compare **Switch-base-8** router assignments against a **T5-base + K-Means** baseline to measure alignment between expert/cluster assignments and coarse POS-based semantic categories.

## Key Findings (N=8)

| Model | ARI | V-Measure | Avg Entropy | Load CV |
|---|---|---|---|---|
| T5-base + K-Means | 0.337 | 0.411 | 1.789 | 0.606 |
| Switch-base-8 | 0.144 | 0.233 | 1.872 | 0.725 |

The T5 baseline with post-hoc clustering shows stronger semantic alignment than Switch's learned routing, suggesting the router optimizes for computational load distribution rather than linguistic specialization.

## Setup

```bash
conda env create -f environment.yml
conda activate moe-caption
python -m spacy download en_core_web_sm
```

## Pipeline

```bash
# 1. Build T5 token table (encodes 10k COCO captions, aligns subwords to spaCy POS labels)
python -m scripts.build_token_table

# 2. Build Switch token table (extracts expert routing for same captions)
python -m scripts.run_switch_inference

# 3. Analyze cluster/expert distributions
python -m scripts.analyze_token_table

# 4. Compute evaluation metrics (ARI, V-Measure, entropy, load CV)
python -m scripts.evaluate_metrics

# 5. Generate heatmap visualizations
python -m scripts.generate_plots
```

## Project Structure

- `src/utils/labeling.py` — spaCy POS → coarse semantic category mapping
- `scripts/build_token_table.py` — T5 encoder token alignment pipeline
- `scripts/run_switch_inference.py` — Switch Transformer expert routing extraction
- `scripts/analyze_token_table.py` — cluster/expert distribution analysis
- `scripts/evaluate_metrics.py` — quantitative evaluation (ARI, V-Measure, entropy, load balance)
- `scripts/generate_plots.py` — heatmap visualization generation
- `artifacts/` — saved token tables, metrics, and plots
