import json
import sys
from itertools import product
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import argparse
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model_args import Args
from src.models.load import load_model
from arrange import arrange_images_on_canvas
from PIL import Image
from src.models.embedder import mean_pooling
@dataclass
class ItemInfo:
    image_path: str  # Keep path for JSON output
    image_data: np.ndarray  # Store preloaded image data
    category: str
    description: str
    image_features: None
    input_ids: None
    attention_mask: None


class BatchedOutfitDataset(Dataset):
    def __init__(self, outfit_combinations, input_processor):
        self.outfit_combinations = outfit_combinations
        self.input_processor = input_processor

    def __len__(self):
        return len(self.outfit_combinations)

    def __getitem__(self, idx):
        outfit_items = self.outfit_combinations[idx]
        images = []
        categories = []
        descriptions = []
        image_features = []
        input_ids = []
        attention_mask = []
        for item in outfit_items:
            # Use preloaded image data directly
            images.append(item.image_data)
            categories.append(item.category)
            if item.description:
                descriptions.append(item.description)
            image_features.append(item.image_features)
            input_ids.append(item.input_ids)
            attention_mask.append(item.attention_mask)


        inputs = self.input_processor(
            category=categories,
            images=images,
            texts=descriptions if descriptions else None,
            do_pad=True
        )
        inputs['image_features'] = image_features
        inputs['input_ids'] = input_ids
        inputs['attention_mask'] = attention_mask

        for k in list(inputs.keys()):
            if len(inputs[k]) > 0:
                pad_item = torch.zeros_like(inputs[k][-1]) if k != 'mask' else torch.BoolTensor([True])
                inputs[k] += [pad_item for _ in range(self.input_processor.outfit_max_length - len(outfit_items))]
                inputs[k] = torch.stack(inputs[k])
            else:
                del inputs[k]

        inputs['mask'] = inputs['mask'].squeeze(1)

        return inputs


def calculate_difference(outfit1, outfit2):
    """Calculate the difference between two outfits as a ratio of differing items."""
    set1 = set(outfit1)
    set2 = set(outfit2)
    total_items = len(set1 | set2)
    differing_items = len(set1 ^ set2)
    return differing_items / total_items if total_items > 0 else 0


class OutfitCombinator:
    def __init__(self, images_dir: str, metadata_file: str, model, input_processor, batch_size: int, device):
        self.images_dir = Path(images_dir)
        self.metadata_file = Path(metadata_file)
        self.model = model
        self.input_processor = input_processor
        self.device = device
        self.batch_size = batch_size
        self.category_mapping = {
            "<all-body>": [],
            "<tops>": [],
            "<bottoms>": [],
            "<shoes>": [],
            "<outerwear>": [],
            "<hats>": [],
            "<scarves>": [],
            "<accessories>": [],
            "<bags>": [],
            "<jewellery>": []
        }
        self.count = 0
        self.number_of_total_combinations = 0

    def load_items(self) -> None:
        """Load and preprocess all items once during initialization"""
        with open(self.metadata_file) as f:
            metadata = json.load(f)

        for image_path, info in metadata.items():
            abs_image_path = self.images_dir / Path(image_path).name

            img = cv2.imread(str(abs_image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            image_features = self.input_processor.image_processor(img)['pixel_values'][0]
            if isinstance(image_features, np.ndarray):
                    image_features = torch.from_numpy(image_features)

            text_ = self.input_processor.text_tokenizer([info["descriptor"]], max_length=self.input_processor.text_max_length, padding=self.input_processor.text_padding, truncation=self.input_processor.text_truncation, return_tensors='pt')
            input_ids = text_['input_ids'].squeeze(0)
            attention_mask = text_['attention_mask'].squeeze(0)


            if info["main_category"] in self.category_mapping:
                item = ItemInfo(
                    image_path=str(abs_image_path),
                    image_data=img,
                    category=info["main_category"],
                    description=info["descriptor"],
                    image_features=image_features,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                self.category_mapping[info["main_category"]].append(item)
        tops = len(self.category_mapping["<tops>"])
        bottoms = len(self.category_mapping["<bottoms>"])
        shoes = len(self.category_mapping["<shoes>"])
        allbodies = len(self.category_mapping["<all-body>"])
        outerwears = len(self.category_mapping["<outerwear>"])
        hats = len(self.category_mapping["<hats>"])
        scarves = len(self.category_mapping["<scarves>"])
        accessories = len(self.category_mapping["<accessories>"])
        bags = len(self.category_mapping["<bags>"])
        jewellery = len(self.category_mapping["<jewellery>"])

        self.number_of_total_combinations = (tops * bottoms * shoes + allbodies * shoes) * (outerwears + 1) * (
                    hats + 1) * (accessories + 1) * (bags + 1) * (jewellery + 1) * (scarves + 1)

    def get_all_subsets_by_category(self) -> List[List[ItemInfo]]:
        """Generate all possible subsets of additional items with at most one item per category."""
        grouped_items = [
            self.category_mapping[category] for category in
            ["<outerwear>", "<hats>", "<scarves>", "<accessories>", "<bags>", "<jewellery>"]
        ]

        category_combinations = [
            [None] + items for items in grouped_items
        ]

        all_combinations = product(*category_combinations)

        subsets = [
            [item for item in combination if item is not None]
            for combination in all_combinations
        ]
        return subsets

    def score_outfits_batch(self, outfit_combinations: List[List[ItemInfo]]) -> List[float]:
        """Score multiple outfit combinations in batches."""
        dataset = BatchedOutfitDataset(outfit_combinations, self.input_processor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        scores = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs = {key: value.to(self.device) for key, value in batch.items()}
                input_embeddings = self.model.batch_encode(inputs)
                probs = self.model.get_score(input_embeddings)
                scores.extend(probs.cpu().tolist())
                self.count += len(probs)
                print(f"Processing combination #{self.count} out of {self.number_of_total_combinations}")

        return scores

    def process_batch(self, combinations_batch: List[List[ItemInfo]]) -> List[Dict]:
        """Process a batch of combinations using the model."""
        scores = self.score_outfits_batch(combinations_batch)
        results = []

        for combination, score in zip(combinations_batch, scores):
            result = {
                "score": score,
                "items": [item.image_path for item in combination],
                "categories": [item.category for item in combination],
                "descriptions": [item.description for item in combination]
            }
            results.append(result)

        return results

    def generate_outfits(self, output_file: str, max_workers: int = 4, topk: int = 3) -> None:
        """Generate and score all valid outfit combinations, ensuring top k results are sufficiently different."""
        additional_subsets = self.get_all_subsets_by_category()
        all_combinations = []

        # Generate all combinations
        for all_body in self.category_mapping["<all-body>"]:
            for shoes in self.category_mapping["<shoes>"]:
                base_outfit = [all_body, shoes]
                for additional_subset in additional_subsets:
                    all_combinations.append(base_outfit + additional_subset)

        for top in self.category_mapping["<tops>"]:
            for bottom in self.category_mapping["<bottoms>"]:
                for shoes in self.category_mapping["<shoes>"]:
                    base_outfit = [top, bottom, shoes]
                    for additional_subset in additional_subsets:
                        all_combinations.append(base_outfit + additional_subset)

        # Split combinations into batches
        batches = [all_combinations[i:i + self.batch_size]
                   for i in range(0, len(all_combinations), self.batch_size)]
        all_results = []

        # Process batches using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_batch, batch) for batch in batches]

            for future in as_completed(futures):
                batch_results = future.result()
                all_results.extend(batch_results)


        # Sort results by score
        all_results.sort(key=lambda x: x["score"], reverse=True)

        # Filter top k with at least 60% difference
        selected_outfits = []
        for result in all_results:
            if result["score"][0] < 0.89:
                print(f"There are only {len(selected_outfits)} outfit combinations with distinct items")
                break
            if len(selected_outfits) >= topk:
                break
            if all(calculate_difference(result["items"], selected["items"]) >= 0.8 for selected in selected_outfits):
                selected_outfits.append(result)

        json_name = output_file + ".json"

        # Save results
        with open(json_name, 'w') as f:
            json.dump({
                "total_combinations": len(all_results),
                "top_k": topk,
                "outfits": selected_outfits
            }, f, indent=2)

        image_name_count = 1
        for result in selected_outfits:
            images = [Image.open(path) for path in result["items"]]
            arranged = arrange_images_on_canvas(images, result["categories"])
            arranged.save(output_file + str(image_name_count) + ".png")
            image_name_count += 1


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description="Generate and score outfit combinations")
    parser.add_argument("--images_dir", required=True, help="Directory containing all the images")
    parser.add_argument("--metadata", required=True, help="Path to the metadata JSON file")
    parser.add_argument("--output", required=True, help="Output file name")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--topk", type=int, default=3, help="Number of top outfits to output")
    args = parser.parse_args()

    # Initialize model and processor
    model_args = Args()
    model_args.checkpoint_dir = './checkpoints'
    # model_args.model_path = './checkpoints/cp/cp_auc0.91.pth'
    model_args.model_path = 'lookify-ml/checkpoints/cp/cp_auc0.91.pth'
    model_args.with_cuda = torch.cuda.is_available()

    model, input_processor = load_model(model_args)
    device = torch.device("cuda" if model_args.with_cuda else "cpu")
    model.to(device)

    # Create and run the outfit combinator
    combinator = OutfitCombinator(
        args.images_dir,
        args.metadata,
        model,
        input_processor,
        batch_size=args.batch_size,
        device=device
    )
    combinator.load_items()
    combinator.generate_outfits(args.output, args.max_workers, args.topk)

    print("Found best outfit in ", (time.time() - start) / 60, "minutes")


if __name__ == "__main__":
    main()