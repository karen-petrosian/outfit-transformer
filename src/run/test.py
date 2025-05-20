import os
import json
import random
import torch
import numpy as np
from itertools import combinations
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Adjust import paths as appropriate:
from ..data.datasets import polyvore
from ..models.load import load_model
from ..data import collate_fn
from ..data.datatypes import FashionCompatibilityQuery, FashionItem


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--polyvore_dir', type=str, default='./datasets/polyvore',
                        help="Root directory of the Polyvore dataset.")
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'],
                        default='nondisjoint', help="Which Polyvore split to use.")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to your trained model checkpoint.")
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip', help="Model type to load.")
    parser.add_argument('--k', type=int, default=5,
                        help="Number of random item IDs to pick from test.json.")
    parser.add_argument('--max_subset_size', type=int, default=6,
                        help="Maximum size of each subset to consider.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for inference.")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="Number of workers for dataloader.")
    return parser.parse_args()


def all_subsets_up_to_size(items, max_r=5):
    """
    Generate all non-empty subsets of 'items' with subset size up to 'max_r'.
    items: list or set
    max_r: int, maximum subset size
    """
    items = list(items)  # ensure indexable
    subsets = []
    for r in range(3, min(len(items), max_r) + 1):
        subsets.extend(combinations(items, r))
    return subsets


class PolyvoreCustomSubsetDataset:
    def __init__(self, dataset_dir, metadata, embedding_dict, subsets):
        self.dataset_dir = dataset_dir
        self.metadata = metadata
        self.embedding_dict = embedding_dict
        self.subsets = subsets

    def __len__(self):
        return len(self.subsets)

    def __getitem__(self, idx):
        subset_ids = self.subsets[idx]

        # Create fashion items for each ID in the subset
        outfit = []
        for item_id in subset_ids:
            item = polyvore.load_item(
                self.dataset_dir,
                self.metadata,
                item_id,
                load_image=False,
                embedding_dict=self.embedding_dict
            )
            outfit.append(item)

        # Return in the same format as PolyvoreCompatibilityDataset.__getitem__
        return {
            'label': 1,  # Dummy label for compatibility
            'query': FashionCompatibilityQuery(outfit=outfit)
        }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -------------------------------------------------------------------------
    # 1. Load test.json for the 'compatibility' task
    # -------------------------------------------------------------------------
    test_json_path = os.path.join(
        args.polyvore_dir,
        args.polyvore_type,
        'compatibility',
        'test.json'
    )
    if not os.path.exists(test_json_path):
        raise FileNotFoundError(f"test.json not found at {test_json_path}")

    with open(test_json_path, 'r') as f:
        test_data = json.load(f)

    # -------------------------------------------------------------------------
    # 2. Collect all item IDs from the test set
    # -------------------------------------------------------------------------
    all_test_ids = set()
    for entry in test_data:
        all_test_ids.update(entry['question'])

    all_test_ids = list(all_test_ids)
    print(f"[Info] Found {len(all_test_ids)} unique item IDs in test.json")

    # -------------------------------------------------------------------------
    # 3. Pick a random subset of k distinct item IDs from the test set
    # -------------------------------------------------------------------------
    if args.k > len(all_test_ids):
        print(f"[Warning] 'k' is larger than available test IDs; reducing to {len(all_test_ids)}")
        args.k = len(all_test_ids)
    random_ids = random.sample(all_test_ids, args.k)
    print(f"[Info] Chosen {len(random_ids)} random IDs: {random_ids}")

    # -------------------------------------------------------------------------
    # 4. Generate all subsets (size = 1..max_subset_size) of these k IDs
    # -------------------------------------------------------------------------
    subsets = all_subsets_up_to_size(random_ids, args.max_subset_size)
    print(f"[Info] Total # of subsets up to size {args.max_subset_size}: {len(subsets)}")

    # -------------------------------------------------------------------------
    # 5. Load metadata and embedding dictionary
    # -------------------------------------------------------------------------
    metadata = polyvore.load_metadata(args.polyvore_dir)
    embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)

    # -------------------------------------------------------------------------
    # 6. Create custom dataset for our subsets
    # -------------------------------------------------------------------------
    custom_dataset = PolyvoreCustomSubsetDataset(
        dataset_dir=args.polyvore_dir,
        metadata=metadata,
        embedding_dict=embedding_dict,
        subsets=subsets
    )

    # Create dataloader using the same collate function as validation
    dataloader = DataLoader(
        custom_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn.cp_collate_fn
    )

    # -------------------------------------------------------------------------
    # 7. Load model
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint)
    model.eval().to(device)

    # -------------------------------------------------------------------------
    # 8. Run inference on each subset
    # -------------------------------------------------------------------------
    results = []

    with torch.no_grad():
        # Use tqdm to show progress bar, just like in validation
        pbar = tqdm(dataloader, desc="[Test] Subset Compatibility")

        for i, data in enumerate(pbar):
            subset_ids = subsets[i * args.batch_size:min((i + 1) * args.batch_size, len(subsets))]

            try:
                # Move labels to device - we use dummy labels
                labels = torch.tensor(data['label'], dtype=torch.float32, device=device)

                # Forward pass - exactly like in validation function
                preds = model(data['query'], use_precomputed_embedding=True).squeeze(1)

                # Store results
                for j, (subset, score) in enumerate(zip(subset_ids, preds.cpu().numpy())):
                    results.append((subset, float(score)))

                # Update progress bar
                if i % 10 == 0 or i == len(dataloader) - 1:
                    pbar.set_postfix({"processed": f"{min((i + 1) * args.batch_size, len(subsets))}/{len(subsets)}"})

            except Exception as e:
                print(f"[Error] Processing subset at batch {i}: {str(e)}")
                continue

    # -------------------------------------------------------------------------
    # 9. Print and save results
    # -------------------------------------------------------------------------
    # Sort results by score
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n[Results] Top 10 Most Compatible Subsets:")
    for subset_ids, score in results[:10]:
        print(f"  {subset_ids} => {score:.4f}")

    print("\n[Results] Bottom 10 Least Compatible Subsets:")
    for subset_ids, score in results[-10:]:
        print(f"  {subset_ids} => {score:.4f}")

    # Save all results to a JSON file
    output_file = f"subset_scores_{args.polyvore_type}_{args.k}items.json"
    results_dict = [{"subset": list(s), "score": float(sc)} for s, sc in results]
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n[Info] All results saved to {output_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()