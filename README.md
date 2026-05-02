# moe-caption-specialization

Analyzing whether MoE (Mixture-of-Experts) routers learn linguistically meaningful expert specialization when encoding natural language text.

## Research Question

When a Switch Transformer encodes text, do its experts naturally specialize along linguistic dimensions (e.g., nouns → expert A, verbs → expert B)? Or is routing driven primarily by load balancing rather than semantic structure?

We compare **Switch-base-8** router assignments against a **T5-base + K-Means** baseline to measure alignment between expert/cluster assignments and POS-based semantic categories at two granularities:

- **Coarse** (5 classes): object, attribute, relation, action, functional
- **Fine-grained** (9 classes): noun, proper_noun, pronoun, adjective, verb, auxiliary, relation, adverb, functional

## Key Findings

### Feature-Budget Fairness Benchmark (N=8, Coarse)

| Feature | Dim | Accuracy | Macro-F1 | ARI |
|---|---|---|---|---|
| Random baseline | - | 20.0% | - | - |
| Majority baseline | - | 34.4% | 0.102 | - |
| Switch hard expert ID | 8 | 71.9% | 0.576 | 0.343 |
| T5 PCA-8D | 8 | 74.7% | 0.599 | 0.329 |
| **Switch all-layer PCA-8D** | **8** | **78.6%** | **0.710** | **0.389** |
| T5 PCA-32D | 32 | 75.2% | 0.615 | 0.341 |
| T5 768D (upper bound) | 768 | 78.4% | 0.726 | 0.366 |
| **Switch all-layer 48D** | **48** | **78.9%** | **0.750** | **0.406** |

At equal dimensionality (8D), Switch all-layer routing outperforms T5 PCA by +3.9% accuracy. With all 48 dimensions, Switch routing matches the T5 768D upper bound while using 16x fewer features. This suggests MoE routing provides a compact, linguistically meaningful signal.

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

# 2. Build Switch token table (extracts all-layer expert routing for same captions)
python -m scripts.run_switch_inference

# 3. Analyze cluster/expert distributions
python -m scripts.analyze_token_table

# 4. Compute evaluation metrics (accuracy, F1, ARI, V-Measure, entropy, purity, load CV)
#    Runs coarse + fine-grained evaluation and feature-budget fairness benchmark
python -m scripts.evaluate_metrics

# 5. Generate heatmap and load balance visualizations
python -m scripts.generate_plots

# 6. Generate summary report
python -m scripts.generate_summary
```

## Project Structure

- `src/utils/labeling.py` — spaCy POS → coarse (5-class) and fine-grained (9-class) category mapping
- `scripts/build_token_table.py` — T5 encoder token alignment pipeline
- `scripts/run_switch_inference.py` — Switch Transformer all-layer expert routing extraction
- `scripts/analyze_token_table.py` — cluster/expert distribution analysis
- `scripts/evaluate_metrics.py` — quantitative evaluation with fairness benchmark
- `scripts/generate_plots.py` — heatmap and load balance visualization
- `scripts/generate_summary.py` — auto-generated findings report
- `artifacts/` — saved token tables, metrics, plots, and reports
