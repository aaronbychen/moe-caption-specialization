# moe-caption-specialization

Analyzing whether MoE (Mixture-of-Experts) routers learn linguistically meaningful expert specialization when encoding natural language text.

## Research Question

When a Switch Transformer encodes text, do its experts naturally specialize along linguistic dimensions (e.g., nouns → expert A, verbs → expert B)? Or is routing driven primarily by load balancing rather than semantic structure?

We compare **Switch-base-8** router assignments against a **T5-base + K-Means** baseline to measure alignment between expert/cluster assignments and POS-based semantic categories at two granularities:

- **Coarse** (5 classes): object, attribute, relation, action, functional
- **Fine-grained** (9 classes): noun, proper_noun, pronoun, adjective, verb, auxiliary, relation, adverb, functional

## Key Findings

### Feature-Budget Fairness Benchmark (N=8, Coarse)

All results evaluated on a held-out test set (caption-level 80/20 split). PCA, KMeans, and majority mapping are fit on train only.

| Feature | Dim | Accuracy | Macro-F1 | ARI |
|---|---|---|---|---|
| Random baseline | - | 20.0% | - | - |
| Majority baseline | - | 35.4% | 0.105 | - |
| Switch hard expert ID | 8 | 72.4% | 0.581 | 0.341 |
| T5 PCA-8D | 8 | 75.6% | 0.610 | 0.343 |
| **Switch all-layer PCA-8D** | **8** | **81.2%** | **0.769** | **0.376** |
| T5 PCA-32D | 32 | 75.5% | 0.606 | 0.334 |
| T5 768D KMeans baseline | 768 | 76.9% | 0.695 | 0.430 |
| **Switch all-layer 48D** | **48** | **81.7%** | **0.670** | **0.430** |
| Word identity baseline | - | 94.8% | 0.929 | 0.880 |

Under a dimensionality-matched unsupervised comparison, Switch all-layer routing features outperform T5 PCA at equal dimensionality (+5.6% accuracy at 8D). With all 48 dimensions, Switch routing outperforms the T5 768D KMeans baseline while using 16x fewer features.

The word identity baseline (94.8%) shows that POS category is largely determined by lexical identity. The routing features capture substantial lexical-syntactic structure, though they do not exceed a direct lexical lookup.

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
