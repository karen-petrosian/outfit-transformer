import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize outfit subsets with their compatibility scores')
    parser.add_argument('--json_file', type=str, required=True,
                        help='Path to the JSON file with subset scores')
    parser.add_argument('--polyvore_dir', type=str, default='./datasets/polyvore',
                        help='Root directory of the Polyvore dataset')
    parser.add_argument('--output_dir', type=str, default='./outfit_visualizations',
                        help='Directory to save visualization images')
    parser.add_argument('--image_size', type=int, default=300,
                        help='Size of each item image (width and height)')
    parser.add_argument('--font_size', type=int, default=20,
                        help='Font size for the score text')
    parser.add_argument('--padding', type=int, default=0,
                        help='Padding between images')
    return parser.parse_args()

def load_image(image_path, target_size):
    """Load and resize an image, maintaining aspect ratio"""
    try:
        img = Image.open(image_path)
        # Resize maintaining aspect ratio
        img.thumbnail((target_size, target_size))

        # Create a blank canvas with the target size and paste the image onto it
        new_img = Image.new('RGB', (target_size, target_size), color=(255, 255, 255))

        # Center the image
        x_offset = (target_size - img.width) // 2
        y_offset = (target_size - img.height) // 2
        new_img.paste(img, (x_offset, y_offset))

        return new_img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Return a blank image with error text
        blank = Image.new('RGB', (target_size, target_size), color=(200, 200, 200))
        draw = ImageDraw.Draw(blank)
        draw.text((10, target_size//2), "Image Error", fill=(0, 0, 0))
        return blank

def visualize_subsets(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the JSON file with subset scores
    with open(args.json_file, 'r') as f:
        subsets_data = json.load(f)

    print(f"Processing {len(subsets_data)} subsets...")

    # For each subset in the JSON file
    for entry in tqdm(subsets_data):
        subset = entry['subset']
        score = entry['score']
        score_rounded = round(score, 3)

        # Load images for each item in the subset
        images = []
        for item_id in subset:
            image_path = os.path.join(args.polyvore_dir, 'images', f"{item_id}.jpg")
            img = load_image(image_path, args.image_size)
            images.append(img)

        # Calculate the width of the combined image
        total_width = len(images) * args.image_size + (len(images) - 1) * args.padding
        max_height = args.image_size

        # Create a new image to hold the horizontally concatenated images
        combined_img = Image.new('RGB', (total_width, max_height + 30), (255, 255, 255))

        # Paste each image side by side
        x_offset = 0
        for img in images:
            combined_img.paste(img, (x_offset, 0))
            x_offset += img.width + args.padding

        # Add the score text at the bottom
        draw = ImageDraw.Draw(combined_img)
        try:
            # Try to load a font, fall back to default if not available
            font = ImageFont.truetype("arial.ttf", args.font_size)
        except:
            font = ImageFont.load_default()

        # Draw score text centered at the bottom
        score_text = f"Score: {score_rounded:.3f}"
        text_width = draw.textlength(score_text, font=font)
        draw.text(
            ((total_width - text_width) // 2, max_height + 5),
            score_text,
            fill=(0, 0, 0),
            font=font
        )

        # Create filename: score_item1_item2_item3.jpg
        subset_str = '_'.join(map(str, subset))
        filename = f"{score_rounded:.3f}_{subset_str}.jpg"
        output_path = os.path.join(args.output_dir, filename)

        # Save the image
        combined_img.save(output_path)

    print(f"All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    visualize_subsets(args)