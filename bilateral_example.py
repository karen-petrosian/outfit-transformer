import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import os

def create_bilateral_concat(image_path, save_path="bilateral_concat.png", diameter=75, sigma_color=75, sigma_space=75):
    # Load image using OpenCV
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    # Apply bilateral filter
    filtered = cv2.bilateralFilter(original, diameter, sigma_color, sigma_space)
    # Convert BGR to RGB for PIL compatibility
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    # Convert to PIL images
    orig_img = Image.fromarray(original_rgb)
    filt_img = Image.fromarray(filtered_rgb)
    # Stack images horizontally
    widths, heights = zip(*(img.size for img in [orig_img, filt_img]))
    total_width = sum(widths)
    max_height = max(heights)
    concat_img = Image.new('RGB', (total_width, max_height))
    concat_img.paste(orig_img, (0, 0))
    concat_img.paste(filt_img, (orig_img.width, 0))
    # Save and show
    concat_img.save(save_path)
    # concat_img.show()

def stack_images_by_ids(ids, images_dir="datasets/polyvore/images", output_color="id_stack_color.png", output_gray="id_stack_gray.png"):
    """
    Given a list of integer IDs, locate corresponding JPG images in `images_dir`,
    stack the color versions horizontally into `output_color`, and the grayscale
    versions horizontally into `output_gray`.
    """
    colored_imgs = []
    gray_imgs = []
    for id in ids:
        filename = f"{id}.jpg"
        img_path = os.path.join(images_dir, filename)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image for ID {id} not found at {img_path}")
        img = Image.open(img_path).convert("RGB")
        colored_imgs.append(img)
        gray_imgs.append(img.convert("L"))

    # Stack color images
    widths, heights = zip(*(img.size for img in colored_imgs))
    total_width = sum(widths)
    max_height = max(heights)
    color_stack = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in colored_imgs:
        color_stack.paste(img, (x_offset, 0))
        x_offset += img.width
    color_stack.save(output_color)
    color_stack.show()

    # Stack grayscale images
    widths_g, heights_g = zip(*(img.size for img in gray_imgs))
    total_width_g = sum(widths_g)
    max_height_g = max(heights_g)
    gray_stack = Image.new("L", (total_width_g, max_height_g))
    x_offset = 0
    for img in gray_imgs:
        gray_stack.paste(img, (x_offset, 0))
        x_offset += img.width
    gray_stack.save(output_gray)
    gray_stack.show()