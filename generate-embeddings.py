#!/usr/bin/env python3
import os
import json
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def main():
    input_dir = Path("/Users/ultihash/test/landscapes_test")
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.")
        return

    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [p for p in input_dir.glob("*") if p.suffix.lower() in supported_extensions]
    if not image_files:
        print("No image files found.")
        return

    print(f"Found {len(image_files)} images in {input_dir}.")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    metadata = []
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        inputs = processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        embedding = image_features[0].tolist()

        metadata.append({
            "filename": image_path.name,
            "embedding": embedding
        })

    output_file = Path("/Users/ultihash/test/landscapes_metadata.json")
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata for {len(metadata)} images saved to {output_file}.")

if __name__ == "__main__":
    main()