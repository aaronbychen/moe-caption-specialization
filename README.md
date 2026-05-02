# moe-caption-specialization

Analyzing whether MoE (Mixture-of-Experts) routers learn linguistically meaningful expert specialization when encoding natural language text.

## Research Question

When a Switch Transformer encodes text, do its experts naturally specialize along linguistic dimensions (e.g., nouns → expert A, verbs → expert B)? Or is routing driven primarily by load balancing, lexical identity, and syntactic regularities?

We compare **Switch-base-8** router assignments against a **T5-base + K-Means** baseline to measure alignment between expert/cluster assignments and POS-derived linguistic categories at two granularities:

- **Coarse** (5 classes): object, attribute, relation, action, functional
- **Fine-grained** (9 classes): noun, proper_noun, pronoun, adjective, verb, auxiliary, relation, adverb, functional

## Key Findings

### Feature-Budget Fairness Benchmark (N=8, Coarse)

All results evaluated on a held-out test set (caption-level 80/20 split). PCA, KMeans, and majority mapping are fit on train only.

| Feature | Dim | Accuracy | Macro-F1 | ARI |
|---|---|---|---|---|
| Random baseline | - | 20.0% | - | - |
| Majority baseline | - | 34.6% | 0.103 | - |
| Switch hard expert ID | 8 | 71.9% | 0.579 | 0.334 |
| T5 PCA-8D | 8 | 75.3% | 0.602 | 0.347 |
| **Switch all-layer PCA-8D** | **8** | **80.7%** | **0.659** | **0.416** |
| T5 PCA-32D | 32 | 75.6% | 0.607 | 0.349 |
| T5 768D KMeans baseline | 768 | 72.3% | 0.468 | 0.327 |
| **Switch all-layer 48D** | **48** | **82.1%** | **0.801** | **0.441** |
| Word identity baseline | - | 97.6% | 0.968 | 0.946 |

Under a dimensionality-matched unsupervised comparison, Switch all-layer routing features outperform T5 PCA at equal dimensionality (+5.4% accuracy at 8D). Switch all-layer 48D now beats T5 768D KMeans on all metrics (accuracy, macro-F1, ARI) while using 16x fewer features. The macro-F1 gap between Switch PCA-8D (0.659) and Switch 48D (0.801) reflects the 48D model's ability to cover all 5 coarse classes including the minority attribute class, while PCA-8D misses it — illustrating why we report both metrics.

The word identity baseline (97.6%) shows that POS category is largely determined by lexical identity. The routing features capture substantial lexical-syntactic structure, though they do not exceed a direct lexical lookup and should be interpreted as compact lexical-syntactic signals rather than evidence of deep semantic understanding.

### 3-Seed Robustness (Coarse, mean±std)

| Feature | Dim | Accuracy | Macro-F1 |
|---|---|---|---|
| Switch all-layer PCA-8D | 8 | 80.5%±0.6 | 0.691±0.055 |
| T5 PCA-8D | 8 | 75.8%±0.8 | 0.608±0.006 |
| Switch all-layer 48D | 48 | 81.5%±0.2 | 0.668±0.003 |
| T5 768D KMeans baseline | 768 | 76.6%±0.8 | 0.640±0.040 |

Results are consistent across 3 caption-level random splits (seeds 42, 123, 7).

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
