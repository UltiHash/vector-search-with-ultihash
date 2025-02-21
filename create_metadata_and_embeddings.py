#!/usr/bin/env python3
import os
import json
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def main():
    # Define the input directory containing your landscape images.
    input_dir = Path("/Users/ultihash/test/landscapes_test")
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.")
        return

    # List only image files with supported extensions.
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [p for p in input_dir.glob("*") if p.suffix.lower() in supported_extensions]
    if not image_files:
        print("No image files found in the directory.")
        return

    print(f"Found {len(image_files)} images in {input_dir}.")

    # Load the CLIP model and processor.
    # We use the 'openai/clip-vit-base-patch32' model for generating image embeddings.
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()  # Set model to evaluation mode

    metadata = []

    # Process each image (using a progress bar for feedback)
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Open the image and ensure it's in RGB mode.
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        # Prepare the image for the model.
        inputs = processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            # Generate the image embedding.
            image_features = model.get_image_features(**inputs)
        
        # Convert the tensor embedding to a list of floats.
        embedding = image_features[0].tolist()
        
        # Create a metadata entry containing the filename and the embedding.
        entry = {
            "filename": image_path.name,
            "embedding": embedding
        }
        metadata.append(entry)

    # Define the output JSON file path.
    output_file = Path("/Users/ultihash/test/landscapes_metadata.json")
    
    # Save the metadata to a JSON file with indentation for readability.
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata for {len(metadata)} images saved to {output_file}.")

if __name__ == "__main__":
    main()

