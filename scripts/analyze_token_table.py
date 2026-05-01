import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import glob
from collections import Counter, defaultdict
from sklearn.cluster import KMeans

def print_summary(label, rows, assign_ids):
    """
    Generic function to print category distribution and top words
    given a list of rows and their corresponding cluster/expert IDs.
    """
    print(f"\n{'='*20} Summary: {label} {'='*20}")
    
    # Organize data
    cluster_to_categories = defaultdict(list)
    cluster_to_words = defaultdict(list)

    for assign_id, row in zip(assign_ids, rows):
        cluster_to_categories[assign_id].append(row["category"])
        cluster_to_words[assign_id].append(row["word"])

    # Print distribution
    unique_ids = sorted(set(assign_ids))
    for aid in unique_ids:
        cat_counts = Counter(cluster_to_categories[aid])
        normalized_words = [w.lower() for w in cluster_to_words[aid]]
        top_words = [w for w, _ in Counter(normalized_words).most_common(10)]
        
        print(f"\n{label} {aid}:")
        print(f"  Count: {len(cluster_to_categories[aid])}")
        print(f"  Distribution: {dict(cat_counts)}")
        print(f"  Top Words: {', '.join(top_words)}")

def main():
    # 1. Load T5 Data (Chunked)
    t5_files = glob.glob("artifacts/aligned_token_table_part*.pt")
    t5_rows = []
    for file in t5_files:
        t5_rows.extend(torch.load(file))
    
    # 2. Load Switch Data (Single File)
    sw_path = "artifacts/switch_token_table_8.pt"
    sw_rows = torch.load(sw_path) if os.path.exists(sw_path) else []

    print(f"Loaded {len(t5_rows)} T5 rows and {len(sw_rows)} Switch rows.")

    # 3. Analyze T5 (K-Means)
    n = 8
    print(f"\n{'='*60}")
    print(f"ANALYZING T5 BASELINE (K-Means N={n})")
    print(f"{'='*60}")
    
    X = torch.stack([row["vector"] for row in t5_rows]).numpy().astype("float32")
    kmeans = KMeans(n_clusters=n, random_state=42, n_init="auto")
    t5_cluster_ids = kmeans.fit_predict(X)
    
    print_summary("T5 Cluster", t5_rows, t5_cluster_ids)

    # 4. Analyze Switch (Direct Expert ID)
    if sw_rows:
        print(f"\n{'='*60}")
        print(f"ANALYZING SWITCH MoE (Expert IDs)")
        print(f"{'='*60}")
        
        sw_expert_ids = [row["expert_id"] for row in sw_rows]
        print_summary("Switch Expert", sw_rows, sw_expert_ids)
    else:
        print("\n[!] Switch data not found, skipping Switch analysis.")

if __name__ == "__main__":
    main()