import torch
from collections import Counter, defaultdict
from sklearn.cluster import KMeans


def main():
    rows = torch.load("artifacts/aligned_token_table.pt")

    print(f"Loaded {len(rows)} aligned rows.")
    print(f"Number of aligned tokens: {len(rows)}")
    print(f"Number of unique captions: {len(set(row['caption_id'] for row in rows))}")

    # category counts
    category_counts = Counter(row["category"] for row in rows)
    print("\nCategory counts:")
    for category, count in category_counts.items():
        print(f"{category}: {count}")

    # stack vectors
    X = torch.stack([row["vector"] for row in rows]).numpy()
    categories = [row["category"] for row in rows]
    words = [row["word"] for row in rows]

    # toy clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X)

    cluster_to_categories = defaultdict(list)
    cluster_to_words = defaultdict(list)

    for cluster_id, category, word in zip(cluster_ids, categories, words):
        cluster_to_categories[cluster_id].append(category)
        cluster_to_words[cluster_id].append(word)

    print("\nCluster summaries:")
    for cluster_id in sorted(cluster_to_categories.keys()):
        cat_counts = Counter(cluster_to_categories[cluster_id])
        print(f"\nCluster {cluster_id}:")
        print("Category distribution:", dict(cat_counts))
        print("Sample words:", cluster_to_words[cluster_id][:10])


if __name__ == "__main__":
    main()