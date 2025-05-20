import os
import numpy as np
import torch
import clip
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import argparse


def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def extract_clip_embedding(image_path, model, preprocess, device):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image).squeeze(0)
    return embedding.cpu().numpy()

# ===========================
# Cluster Characterization
# ===========================

def compute_cluster_scores(X, labels, category_map=None):
    n_clusters = np.max(labels) + 1
    scores = []

    for i in range(n_clusters):
        cluster_points = X[labels == i]
        size = len(cluster_points)
        size_ratio = size / len(X)
        cohesion = np.mean(np.linalg.norm(cluster_points - cluster_points.mean(axis=0), axis=1))

        centroid = cluster_points.mean(axis=0)
        dists_to_others = [np.linalg.norm(centroid - X[labels == j].mean(axis=0))
                           for j in range(n_clusters) if j != i]
        isolation = np.mean(dists_to_others)

        if category_map is not None:
            categories = [category_map.get(idx, None) for idx in np.where(labels == i)[0]]
            consistency = 1.0 if len(set(categories)) == 1 else 0.0
        else:
            consistency = 0.0  # Placeholder without labels

        # Weighted composite score
        noise_score = (
            0.3 * (1 - size_ratio) +
            0.2 * cohesion +
            0.2 * isolation +
            0.3 * (1 - consistency)
        )

        scores.append((i, noise_score, size, cohesion, isolation, consistency))

    return sorted(scores, key=lambda x: -x[1])  # descending by noise score

# ===========================
# Main Routine
# ===========================

def main():
    parser = argparse.ArgumentParser(description="K-Means CLIP Noise Detection")
    parser.add_argument("--img_dir", required=True, help="Path to images")
    parser.add_argument("--out_dir", required=True, help="Where to save filtered images")
    parser.add_argument("--k", type=int, default=15, help="Number of clusters")
    parser.add_argument("--threshold", type=float, default=0.65, help="Noise score threshold")
    parser.add_argument("--embedding_cache", default=None, help="Optional .npy file to load/store embeddings")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, preprocess, device = load_clip_model()

    image_paths = sorted([os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir)
                          if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    if args.embedding_cache and os.path.exists(args.embedding_cache):
        embeddings = np.load(args.embedding_cache)
    else:
        from PIL import Image
        embeddings = []
        for img_path in tqdm(image_paths, desc="Extracting CLIP embeddings"):
            embedding = extract_clip_embedding(img_path, model, preprocess, device)
            embeddings.append(embedding)
        embeddings = np.vstack(embeddings)
        if args.embedding_cache:
            np.save(args.embedding_cache, embeddings)


    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    scores = compute_cluster_scores(embeddings, labels)

    print("\nTop clusters by noise score:")
    for cid, score, size, coh, iso, cons in scores[:5]:
        print(f"Cluster {cid}: score={score:.3f}, size={size}, cohesion={coh:.2f}, isolation={iso:.2f}")

    noise_clusters = {cid for cid, score, *_ in scores if score > args.threshold}

    for i, img_path in enumerate(image_paths):
        if labels[i] not in noise_clusters:
            img = cv2.imread(img_path)
            out_path = os.path.join(args.out_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, img)

    print(f"\nSaved {len(image_paths) - len(noise_clusters)} clean images to: {args.out_dir}")

if __name__ == "__main__":
    main()
