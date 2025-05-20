from PIL import Image
import numpy as np
import argparse
import os

def crop_image(image):
    image_data = np.array(image)
    non_transparent_mask = image_data[:, :, 3] > 0
    if np.any(non_transparent_mask):
        non_empty_cols = np.where(non_transparent_mask.any(axis=0))[0]
        non_empty_rows = np.where(non_transparent_mask.any(axis=1))[0]
        crop_box = (min(non_empty_cols), min(non_empty_rows), max(non_empty_cols) + 1, max(non_empty_rows) + 1)
        cropped_image = image.crop(crop_box)
        return cropped_image
    return image


def get_item_width(image):
    image_data = np.array(image)
    non_transparent_mask = image_data[:, :, 3] > 0
    row_widths = []
    for row in non_transparent_mask:
        non_transparent_pixels = np.where(row)[0]
        if len(non_transparent_pixels) > 0:
            width = non_transparent_pixels[-1] - non_transparent_pixels[0]
            row_widths.append(width)
    return np.mean(row_widths) if row_widths else image.width


def resize_image_by_avg_width(image, target_avg_width):
    current_avg_width = get_item_width(image)
    scale_factor = target_avg_width / current_avg_width
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def resize_image(image, width):
    aspect_ratio = image.height / image.width
    new_height = int(aspect_ratio * width)
    return image.resize((width, new_height), Image.Resampling.LANCZOS)


def place_image_centered(base_image, item_image, center_position):
    x_center, y_center = center_position
    left = x_center - item_image.width // 2
    top = y_center - item_image.height // 2

    # Ensure the image does not go beyond canvas boundaries
    left = max(10, min(left, base_image.width - item_image.width - 10))
    top = max(10, min(top, base_image.height - item_image.height - 10))

    base_image.paste(item_image, (left, top), item_image)


def arrange_images_on_canvas(images, categories, canvas_width=600, canvas_height=1100):
    large_width = int(canvas_width * 0.418)
    small_width = large_width // 3
    hat_width = int(large_width * 0.5)

    canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))

    top_width = int(large_width * 0.85)
    outerwear_width = int(top_width * 1.28)

    positions = {
        '<outerwear>': (int(canvas_width * 0.32), int(canvas_height * 0.3)),
        '<tops>': (int(canvas_width * 0.64), int(canvas_height * 0.3)),
        '<bottoms>': (int(canvas_width * 0.53), int(canvas_height * 0.6)),
        '<shoes>': (int(canvas_width * 0.53), int(canvas_height * 0.9)),
        '<mask>': (int(canvas_width * 0.53), int(canvas_height * 0.15)),
        '<scarves>': (int(canvas_width * 0.74), int(canvas_height * 0.3)),
        '<hats>': (int(canvas_width * 0.53), int(canvas_height * 0.1)),
        '<all-body>': (int(canvas_width * 0.53), int(canvas_height * 0.4)),
        '<accessories>': (int(canvas_width * 0.74), int(canvas_height * 0.55)),
        '<jewellery>': (int(canvas_width * 0.84), int(canvas_height * 0.4)),
        '<sunglasses>': (int(canvas_width * 0.74), int(canvas_height * 0.45)),
        '<bags>': (int(canvas_width * 0.28), int(canvas_height * 0.65)),
    }

    ordered_categories = [cat for cat in categories if
                          cat not in ['<bottoms>', '<tops>', '<hats>', '<shoes>', '<bags>']]
    if '<bottoms>' in categories:
        ordered_categories.append('<bottoms>')
    if '<tops>' in categories:
        ordered_categories.append('<tops>')
    if '<hats>' in categories:
        ordered_categories.append('<hats>')
    if '<bags>' in categories:
        ordered_categories.append('<bags>')
    if '<shoes>' in categories:
        ordered_categories.append('<shoes>')
    if '<accessories>' in categories:
        ordered_categories.append('<accessories>')

    for category in ordered_categories:
        img_index = categories.index(category)
        img = crop_image(images[img_index])

        if category in ['<shoes>', '<bags>', '<jewellery>']:
            if category == '<jewellery>':
                img = resize_image(img, small_width)
            else:
                img = resize_image(img, large_width)
        else:
            if category == '<tops>':
                img = resize_image_by_avg_width(img, top_width)
            elif category == '<outerwear>':
                img = resize_image_by_avg_width(img, outerwear_width)
            elif category == '<hats>':
                img = resize_image_by_avg_width(img, hat_width)
            elif category in ['<bottoms>', '<all-body>']:
                img = resize_image_by_avg_width(img, large_width)
            else:
                img = resize_image_by_avg_width(img, small_width)

        if category in positions:
            center_position = positions[category]
            place_image_centered(canvas, img, center_position)

    final_image = canvas.convert('RGB')
    return final_image

