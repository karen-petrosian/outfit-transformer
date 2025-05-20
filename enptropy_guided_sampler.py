#!/usr/bin/env python3
# entropy_guided_sampler.py

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
import json

# =====================================
# Histogram + Entropy Calculation
# =====================================

def compute_cielab_histogram(image_paths, bins=4):
    """Compute 3D histogram in CIELAB space across multiple images."""
    hist = np.zeros((bins, bins, bins), dtype=np.float32)
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        lab = lab / 255.0  # Normalize
        bin_indices = (lab * bins).astype(int).clip(0, bins-1)
        for i in range(lab.shape[0]):
            for j in range(lab.shape[1]):
                l, a, b = bin_indices[i, j]
                hist[l, a, b] += 1
    hist_flat = hist.flatten()
    hist_prob = hist_flat / hist_flat.sum()
    return hist_prob

def shannon_entropy(hist_prob):
    """Compute entropy from a normalized histogram."""
    return -np.sum(hist_prob[hist_prob > 0] * np.log(hist_prob[hist_prob > 0]))

# =====================================
# Entropy Scoring for Dataset
# =====================================

def compute_entropies(outfit_json, img_root, bins=4):
    """Compute entropy per outfit from JSON spec of outfit image paths."""
    entropy_map = {}
    for outfit_id, item_list in tqdm(outfit_json.items(), desc="Computing entropies"):
        image_paths = [os.path.join(img_root, path) for path in item_list]
        hist = compute_cielab_histogram(image_paths, bins=bins)
        entropy = shannon_entropy(hist)
        entropy_map[outfit_id] = entropy
    return entropy_map

# =====================================
# Curriculum Sampling Wrapper
# =====================================

class EntropySampler(torch.utils.data.Sampler):
    def __init__(self, entropy_scores, alpha=1.0):
        self.entropies = np.array([entropy_scores[i] for i in sorted(entropy_scores)])
        self.alpha = alpha
        self.weights = self._compute_weights()

    def _compute_weights(self):
        """Normalize and exponentiate entropies to produce sampling weights."""
        norm = (self.entropies - self.entropies.min()) / (self.entropies.max() - self.entropies.min() + 1e-6)
        weights = norm ** self.alpha
        return weights / weights.sum()

    def update_alpha(self, new_alpha):
        self.alpha = new_alpha
        self.weights = self._compute_weights()

    def __iter__(self):
        return iter(torch.multinomial(torch.tensor(self.weights), len(self.weights), replacement=True).tolist())

    def __len__(self):
        return len(self.weights)

# =====================================
# Example Integration
# =====================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entropy-Guided Sampler Preprocessing")
    parser.add_argument("--json", required=True, help="Outfit JSON file: {outfit_id: [img1, img2, ...]}")
    parser.add_argument("--img_root", required=True, help="Path to all outfit images")
    parser.add_argument("--out", default="entropy_scores.json", help="Where to save entropy scores")
    args = parser.parse_args()

    with open(args.json, "r") as f:
        outfit_json = json.load(f)

    entropy_scores = compute_entropies(outfit_json, args.img_root)
    with open(args.out, "w") as f:
        json.dump(entropy_scores, f, indent=2)

    print(f"\n✅ Saved entropy scores to: {args.out}")
