import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import glob
import math
import torch
import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, v_measure_score

def compute_load_metrics(assignments, n_experts):
    counts = Counter(assignments)
    expert_counts = [counts.get(i, 0) for i in range(n_experts)]
    mean_load = np.mean(expert_counts)
    std_load = np.std(expert_counts)
    load_cv = std_load / mean_load if mean_load > 0 else 0.0
    return expert_counts, load_cv

def compute_purity(categories, assignments, n_experts):
    expert_cats = defaultdict(list)
    for cat, assign in zip(categories, assignments):
        expert_cats[assign].append(cat)
    purities = {}
    weighted_sum = 0.0
    total = len(categories)
    for eid in range(n_experts):
        tokens = expert_cats.get(eid, [])
        if not tokens:
            purities[eid] = 0.0
            continue
        dominant_count = Counter(tokens).most_common(1)[0][1]
        purities[eid] = dominant_count / len(tokens)
        weighted_sum += dominant_count
    avg_purity = weighted_sum / total if total > 0 else 0.0
    return purities, avg_purity

def compute_entropy_per_category(categories, assignments):
    cat_to_assign = defaultdict(list)
    for cat, assign in zip(categories, assignments):
        cat_to_assign[cat].append(assign)
    entropies = {}
    for cat, assigns in cat_to_assign.items():
        counts = Counter(assigns)
        total = len(assigns)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0: entropy -= p * math.log2(p)
        entropies[cat] = entropy
    avg_entropy = sum(entropies.values()) / len(entropies) if entropies else 0.0
    return entropies, avg_entropy

def eval_model(label, categories, assignments, n):
    ari = adjusted_rand_score(categories, assignments)
    vmeasure = v_measure_score(categories, assignments)
    cat_entropy, avg_entropy = compute_entropy_per_category(categories, assignments)
    counts, load_cv = compute_load_metrics(assignments, n)
    purities, avg_purity = compute_purity(categories, assignments, n)

    print(f"\n[{label}]")
    print(f"  ARI: {ari:.4f} | V-Meas: {vmeasure:.4f} | Entropy: {avg_entropy:.4f} | Purity: {avg_purity:.4f} | Load CV: {load_cv:.4f}")

    return {
        "ari": ari, "v_measure": vmeasure,
        "avg_entropy": avg_entropy, "load_cv": load_cv,
        "avg_purity": avg_purity,
        "per_expert_purity": {str(k): v for k, v in purities.items()},
        "expert_counts": counts
    }

def run_eval(t5_data, t5_vectors, switch_data_map, cat_key, label_suffix):
    """Run evaluation for a given category key (coarse or fine)."""
    t5_categories = [row[cat_key] for row in t5_data]
    results = {}

    for n in [4, 8, 16]:
        print(f"\n{'='*60}")
        print(f"N={n} | {label_suffix}")
        print(f"{'='*60}")

        kmeans = KMeans(n_clusters=n, random_state=42, n_init="auto")
        t5_clusters = kmeans.fit_predict(t5_vectors)
        entry = {"t5_baseline": eval_model(f"T5 K-Means (k={n})", t5_categories, t5_clusters.tolist(), n)}

        sw_data = switch_data_map.get(n)
        if sw_data:
            sw_ids = [row["expert_id"] for row in sw_data]
            sw_cats = [row.get(cat_key, row["category"]) for row in sw_data]
            entry["switch_model"] = eval_model(f"Switch MoE (N={n})", sw_cats, sw_ids, n)

        results[f"n_{n}"] = entry

    return results

def main():
    all_files = glob.glob("artifacts/aligned_token_table_part*.pt")
    t5_data = []
    for file in all_files:
        t5_data.extend(torch.load(file))
    t5_vectors = torch.stack([row["vector"] for row in t5_data]).numpy().astype("float32")

    print(f"Loaded {len(t5_data)} T5 aligned tokens.")

    # check if fine_category is available
    has_fine = "fine_category" in t5_data[0] if t5_data else False
    print(f"Fine-grained categories available: {has_fine}")

    # load all available switch data
    switch_data_map = {}
    for n in [4, 8, 16]:
        sw_path = f"artifacts/switch_token_table_{n}.pt"
        if os.path.exists(sw_path):
            switch_data_map[n] = torch.load(sw_path)

    # coarse evaluation
    print("\n" + "#" * 60)
    print("COARSE CATEGORIES (5 classes)")
    print("#" * 60)
    coarse_results = run_eval(t5_data, t5_vectors, switch_data_map, "category", "coarse")

    output = {"coarse": coarse_results}

    # fine evaluation
    if has_fine:
        print("\n" + "#" * 60)
        print("FINE-GRAINED CATEGORIES (9 classes)")
        print("#" * 60)
        fine_results = run_eval(t5_data, t5_vectors, switch_data_map, "fine_category", "fine")
        output["fine"] = fine_results

    with open("artifacts/eval_metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved metrics to artifacts/eval_metrics.json")

if __name__ == "__main__":
    main()
