import os
import cv2
import torch
import numpy as np
import argparse
from torchvision import transforms
from timm import create_model
from torch.nn.functional import normalize

def load_dino_model():
    """Load pretrained DINO ViT-S/16 model."""
    model = create_model("vit_small_patch16_224_dino", pretrained=True)
    model.eval()
    return model

def preprocess_image(img_path):
    """Preprocess image to feed into DINO."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(img).unsqueeze(0)

def extract_dino_embedding(model, image_tensor):
    """Extract 384-dim DINO embedding from image."""
    with torch.no_grad():
        feats = model.forward_features(image_tensor)
        global_token = feats['x_norm_clstoken']  # [1, 384]
    return normalize(global_token, dim=-1).squeeze(0).cpu().numpy()

def load_prototypes(proto_dir):
    """Load category prototype vectors from a directory."""
    prototypes = {}
    for file in os.listdir(proto_dir):
        if file.endswith('.npy'):
            cat = file.replace('.npy', '')
            prototypes[cat] = np.load(os.path.join(proto_dir, file))
    return prototypes

def compute_cosine_similarity(e1, e2):
    """Compute cosine similarity between two vectors."""
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

def classify_image(embedding, prototypes, threshold=0.6):
    """Classify image by comparing with prototypes."""
    max_sim = -1
    best_cat = None
    for cat, proto in prototypes.items():
        sim = compute_cosine_similarity(embedding, proto)
        if sim > max_sim:
            max_sim = sim
            best_cat = cat
    return best_cat if max_sim >= threshold else "non-garment", max_sim

def filter_images(input_dir, proto_dir, output_dir, threshold=0.6):
    """Process and classify all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    model = load_dino_model()
    prototypes = load_prototypes(proto_dir)

    for file in os.listdir(input_dir):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(input_dir, file)
        try:
            img_tensor = preprocess_image(path)
            embedding = extract_dino_embedding(model, img_tensor)
            label, sim = classify_image(embedding, prototypes, threshold)
            print(f"{file}: {label} (sim={sim:.3f})")

            if label != "non-garment":
                cv2.imwrite(os.path.join(output_dir, file), cv2.imread(path))
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINO-based Garment Filtering")
    parser.add_argument("--input_dir", required=True, help="Directory with input images")
    parser.add_argument("--proto_dir", required=True, help="Directory with prototype vectors (.npy files)")
    parser.add_argument("--output_dir", required=True, help="Where to save garment images")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold for classification")
    args = parser.parse_args()

    filter_images(args.input_dir, args.proto_dir, args.output_dir, args.threshold)
