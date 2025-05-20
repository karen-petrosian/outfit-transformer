import argparse
import json
import os
from PIL import Image
import torch

# Adjust these imports to your project structure
from ..data.datatypes import FashionItem, FashionCompatibilityQuery
from ..models.load import load_model


def load_inputs(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_fashion_items(items):
    fashion_items = []
    for idx, itm in enumerate(items):
        img = Image.open(itm['image']).convert('RGB')
        fashion_items.append(
            FashionItem(
                item_id=idx,
                category=itm.get('category', 'unknown'),
                description=itm.get('description', ''),
                image=img,
                embedding=None
            )
        )
    return fashion_items


def compute_embeddings(fashion_items, model):
    """
    Uses model.precompute_clip_embedding to compute embeddings in batch.
    Handles both torch.Tensor and numpy.ndarray outputs.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model.precompute_clip_embedding(fashion_items)

    # Assign embeddings back, converting to numpy if needed
    for item, emb in zip(fashion_items, embeddings):
        if isinstance(emb, torch.Tensor):
            item.embedding = emb.cpu().numpy()
        else:
            # Already numpy array
            item.embedding = emb
    return fashion_items


def compute_compatibility_score(fashion_items, model):
    model.eval()
    query = FashionCompatibilityQuery(outfit=fashion_items)
    with torch.no_grad():
        pred = model([query], use_precomputed_embedding=True).squeeze(1)
    return pred.item()


def main():
    parser = argparse.ArgumentParser(description="Compute compatibility score using project model.")
    parser.add_argument('--input_json', type=str, required=True, help='Path to JSON list of items')
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'], default='clip')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    args = parser.parse_args()

    items = load_inputs(args.input_json)
    if not items:
        print("No items in input JSON.")
        return

    fashion_items = create_fashion_items(items)
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint)

    fashion_items = compute_embeddings(fashion_items, model)
    score = compute_compatibility_score(fashion_items, model)
    print(f"Compatibility Score: {score:.4f}")

if __name__ == '__main__':
    main()

# good top good bottom 0.98
# good top good bottom bad shoes 0.0844
# good top good bottom good shoes 0.99