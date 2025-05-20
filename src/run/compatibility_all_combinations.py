import json
import os
import pathlib
import random
import itertools
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.datasets import polyvore
from ..models.load import load_model
from ..utils.utils import seed_everything
from ..data.datatypes import FashionItem, FashionCompatibilityQuery

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
RESULT_DIR = SRC_DIR / 'results'
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip')
    parser.add_argument('--polyvore_dir', type=str,
                        default='./datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'],
                        default='nondisjoint')
    parser.add_argument('--batch_sz_per_gpu', type=int,
                        default=512)
    parser.add_argument('--n_workers_per_gpu', type=int,
                        default=4)
    parser.add_argument('--wandb_key', type=str,
                        default=None)
    parser.add_argument('--seed', type=int,
                        default=42)
    parser.add_argument('--checkpoint', type=str,
                        default=None)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of random items to select')
    parser.add_argument('--num_sets', type=int, default=3,
                        help='Number of random sets to generate')
    parser.add_argument('--min_items_in_combination', type=int, default=2,
                        help='Minimum number of items in a combination')
    return parser.parse_args()


# Custom function to load metadata with proper encoding
def load_metadata_safe(polyvore_dir):
    """Load metadata with UTF-8 encoding and error handling."""
    metadata_path = os.path.join(polyvore_dir, "item_metadata.json")
    try:
        # Try with UTF-8 encoding first
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_ = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Fall back to latin-1 which can decode any byte sequence
        with open(metadata_path, 'r', encoding='latin-1') as f:
            metadata_ = json.load(f)
    return metadata_


# Custom function to load embedding dict safely
def load_embedding_dict_safe(polyvore_dir):
    """Load embedding dict with UTF-8 encoding and error handling."""
    embedding_path = os.path.join(polyvore_dir, "precomputed_clip_embeddings/polyvore_0.pkl")
    import pickle
    with open(embedding_path, 'rb') as f:
        embedding_dict = pickle.load(f)
    return embedding_dict


def load_images(item_ids, polyvore_dir):
    """Load images for the given item IDs."""
    images = {}
    for item_id in item_ids:
        img_path = os.path.join(polyvore_dir, "images", f"{item_id}.jpg")
        if os.path.exists(img_path):
            try:
                images[item_id] = Image.open(img_path)
            except Exception as e:
                print(f"Error loading image {item_id}: {e}")
                images[item_id] = Image.new('RGB', (224, 224), color='gray')
        else:
            print(f"Image not found for {item_id}, using placeholder")
            images[item_id] = Image.new('RGB', (224, 224), color='gray')
    return images


def create_combination_image(images, score, item_ids, output_path):
    """Create an image showing items and their compatibility score."""
    n_items = len(item_ids)
    plt.figure(figsize=(n_items * 4, 6))
    # Display images
    for i, item_id in enumerate(item_ids):
        plt.subplot(2, n_items, i + 1)
        plt.imshow(images[item_id])
        plt.title(f"Item {i + 1}: {item_id}")
        plt.axis('off')
    # Display score
    plt.subplot(2, 1, 2)
    plt.text(0.5, 0.5, f"Compatibility Score: {score:.4f}",
             ha='center', va='center', fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def get_item_ids_from_test_set(test_dataset):
    """Extract unique item IDs from the test dataset."""
    all_item_ids = set()
    if hasattr(test_dataset, 'outfit_data'):
        for outfit in test_dataset.outfit_data:
            if 'items' in outfit:
                for item in outfit['items']:
                    if isinstance(item, dict) and 'item_id' in item:
                        all_item_ids.add(item['item_id'])
    if not all_item_ids and hasattr(test_dataset, 'data'):
        for entry in test_dataset.data:
            if isinstance(entry, dict) and 'items' in entry:
                for item in entry['items']:
                    if isinstance(item, dict) and 'item_id' in item:
                        all_item_ids.add(item['item_id'])
    if not all_item_ids and hasattr(test_dataset, 'embedding_dict'):
        all_item_ids = set(test_dataset.embedding_dict.keys())
    print(f"Found {len(all_item_ids)} unique items in test dataset")
    return all_item_ids


def find_item_metadata(item_id, metadata):
    """Find metadata for a specific item ID in a list of metadata."""
    if isinstance(metadata, list):
        for item in metadata:
            if isinstance(item, dict) and item.get('item_id') == item_id:
                return item
        return {}
    elif isinstance(metadata, dict):
        return metadata.get(item_id, {})
    return {}


def create_fashion_item(item_id, metadata, embedding):
    """Create a FashionItem with the specified data."""
    item_info = find_item_metadata(item_id, metadata)
    description = item_info.get('title', f"Item {item_id}")
    category = item_info.get('category_id', "unknown")
    fashion_item = FashionItem(
        item_id=item_id,
        category=category,
        description=description,
        embedding=embedding
    )
    return fashion_item


def run_combinations(args):
    # Load dataset and model
    try:
        metadata = polyvore.load_metadata(args.polyvore_dir)
        embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)
    except Exception as e:
        print(f"Error loading data with original method: {e}")
        print("Trying alternative loading methods...")
        metadata = load_metadata_safe(args.polyvore_dir)
        embedding_dict = load_embedding_dict_safe(args.polyvore_dir)

    if isinstance(metadata, list):
        print(f"Successfully loaded metadata for {len(metadata)} items (list format)")
    else:
        print(f"Successfully loaded metadata for {len(metadata)} items (dict format)")

    print(f"Successfully loaded embeddings for {len(embedding_dict)} items")

    try:
        test = polyvore.PolyvoreCompatibilityDataset(
            dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type,
            dataset_split='test', metadata=metadata, embedding_dict=embedding_dict
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        from types import SimpleNamespace
        test = SimpleNamespace()
        test.metadata = metadata
        test.embedding_dict = embedding_dict

    try:
        model = load_model(model_type=args.model_type, checkpoint=args.checkpoint)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Will skip model prediction and just create visualizations")
        model = None

    # Create output directory
    if args.checkpoint:
        result_dir = os.path.join(
            RESULT_DIR, os.path.basename(os.path.dirname(args.checkpoint)), 'combinations'
        )
    else:
        result_dir = os.path.join(RESULT_DIR, 'compatibility_combinations')
    os.makedirs(result_dir, exist_ok=True)
    print(f"Results will be saved to {result_dir}")

    # Filter valid item IDs to ensure they represent numeric IDs only
    valid_item_ids = {k for k in embedding_dict.keys() if (isinstance(k, int)) or (isinstance(k, str) and k.isdigit())}
    print(f"Found {len(valid_item_ids)} valid items with embeddings")

    # Process multiple random sets
    for set_idx in range(args.num_sets):
        k = min(args.k, len(valid_item_ids))
        if k < args.k:
            print(f"Warning: Only {k} items available, using that instead of requested {args.k}")
        random_items = random.sample(list(valid_item_ids), k)
        print(f"Set {set_idx + 1}: Selected {len(random_items)} random items")
        images = load_images(random_items, args.polyvore_dir)
        set_dir = os.path.join(result_dir, f"set_{set_idx + 1}")
        os.makedirs(set_dir, exist_ok=True)
        min_items = min(args.min_items_in_combination, k)
        all_combinations = []
        for size in range(min_items, k + 1):
            all_combinations.extend(list(itertools.combinations(random_items, size)))
        print(f"Processing {len(all_combinations)} combinations for set {set_idx + 1}")
        for combo_idx, combination in enumerate(tqdm(all_combinations)):
            fashion_items = []
            for item_id in combination:
                if item_id in embedding_dict:
                    # Convert the item_id to integer if it's a numeric string, to avoid pydantic issues
                    if isinstance(item_id, str) and item_id.isdigit():
                        cleaned_item_id = int(item_id)
                    else:
                        cleaned_item_id = item_id
                    fashion_item = create_fashion_item(
                        item_id=cleaned_item_id,
                        metadata=metadata,
                        embedding=embedding_dict[item_id]
                    )
                    fashion_items.append(fashion_item)
            if len(fashion_items) != len(combination):
                continue
            score = 0.0
            if model is not None:
                try:
                    with torch.no_grad():
                        query = FashionCompatibilityQuery(outfit=fashion_items)
                        pred = model([query], use_precomputed_embedding=True).squeeze(1)
                        score = pred.item()
                except Exception as e:
                    print(f"Error getting prediction for combination {combo_idx + 1}: {e}")
                    score = 0.0
            output_path = os.path.join(set_dir, f"combo_{combo_idx + 1}.jpg")
            create_combination_image(images, score, combination, output_path)
        with open(os.path.join(set_dir, 'items.json'), 'w') as f:
            json.dump({"items": list(random_items)}, f)
        print(f"Results for set {set_idx + 1} saved to {set_dir}")


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    run_combinations(args)
