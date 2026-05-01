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
    """
    Computes detailed counts and a single 'load_cv' metric.
    CV = standard deviation / mean. (0 = perfectly balanced)
    """
    counts = Counter(assignments)
    expert_counts = [counts.get(i, 0) for i in range(n_experts)]
    
    mean_load = np.mean(expert_counts)
    std_load = np.std(expert_counts)
    load_cv = std_load / mean_load if mean_load > 0 else 0.0
    
    return expert_counts, load_cv

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

def main():
    all_files = glob.glob("artifacts/aligned_token_table_part*.pt")
    t5_data = []
    for file in all_files:
        t5_data.extend(torch.load(file))
    
    t5_vectors = torch.stack([row["vector"] for row in t5_data])
    t5_categories = [row["category"] for row in t5_data]
    
    print(f"Loaded {len(t5_data)} T5 aligned tokens.")
    
    results = {}

    for n in [8]:
        print(f"\nEvaluating N = {n} Experts/Clusters...")
        
        # --- T5 K-Means ---
        kmeans = KMeans(n_clusters=n, random_state=42, n_init="auto")
        t5_clusters = kmeans.fit_predict(t5_vectors.numpy().astype("float32"))
        
        t5_ari = adjusted_rand_score(t5_categories, t5_clusters)
        t5_vmeasure = v_measure_score(t5_categories, t5_clusters)
        t5_cat_entropy, t5_avg_entropy = compute_entropy_per_category(t5_categories, t5_clusters)
        t5_counts, t5_load_cv = compute_load_metrics(t5_clusters, n)
        
        # --- Switch Model ---
        try:
            # Note: Ensure this file is the one using the same 10k captions for fair comparison
            switch_data = torch.load(f"artifacts/switch_token_table_{n}.pt")
        except FileNotFoundError:
            print(f"  -> Skipping Switch-{n}: File not found.")
            continue
            
        switch_expert_ids = [row["expert_id"] for row in switch_data]
        switch_categories = [row["category"] for row in switch_data]
        
        switch_ari = adjusted_rand_score(switch_categories, switch_expert_ids)
        switch_vmeasure = v_measure_score(switch_categories, switch_expert_ids)
        switch_cat_entropy, switch_avg_entropy = compute_entropy_per_category(switch_categories, switch_expert_ids)
        switch_counts, switch_load_cv = compute_load_metrics(switch_expert_ids, n)
        
        # Store Results
        results[f"n_{n}"] = {
            "t5_baseline": {
                "ari": t5_ari, 
                "v_measure": t5_vmeasure, 
                "avg_entropy": t5_avg_entropy,
                "load_cv": t5_load_cv,
                "expert_counts": t5_counts
            },
            "switch_model": {
                "ari": switch_ari, 
                "v_measure": switch_vmeasure, 
                "avg_entropy": switch_avg_entropy,
                "load_cv": switch_load_cv,
                "expert_counts": switch_counts
            }
        }
        
        # --- Print Detailed Comparison ---
        print(f"\n[T5 Baseline (K-Means)]")
        print(f"  Metrics  -> ARI: {t5_ari:.4f} | V-Meas: {t5_vmeasure:.4f} | Avg Entropy: {t5_avg_entropy:.4f}")
        print(f"  Load     -> CV: {t5_load_cv:.4f} (Perfect=0.0)")
        print(f"  Counts   -> {t5_counts}")
        print("  Category Entropies:")
        for cat, ent in sorted(t5_cat_entropy.items()):
            print(f"    {cat:<12}: {ent:.4f} bits")
            
        print(f"\n[Switch MoE (Router)]")
        print(f"  Metrics  -> ARI: {switch_ari:.4f} | V-Meas: {switch_vmeasure:.4f} | Avg Entropy: {switch_avg_entropy:.4f}")
        print(f"  Load     -> CV: {switch_load_cv:.4f} (Perfect=0.0)")
        print(f"  Counts   -> {switch_counts}")
        print("  Category Entropies:")
        for cat, ent in sorted(switch_cat_entropy.items()):
            print(f"    {cat:<12}: {ent:.4f} bits")
        print("-" * 60)

    with open("artifacts/eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved metrics and load data to artifacts/eval_metrics.json")

if __name__ == "__main__":
    main()