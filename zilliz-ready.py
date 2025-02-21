#!/usr/bin/env python3
import os
import json
from pathlib import Path

def main():
    # Define input and output file paths.
    input_file = Path("/users/ultihash/test/landscapes_metadata.json")
    output_file = Path("/users/ultihash/test/landscape_metadata_col.json")
    
    # Read the original metadata file.
    with open(input_file, "r") as f:
        metadata_list = json.load(f)
    
    # Transform the metadata entries.
    transformed_data = []
    for entry in metadata_list:
        # Remove the .jpg extension from the filename.
        filename = entry.get("filename", "")
        base_filename, _ = os.path.splitext(filename)
        
        transformed_entry = {
            "filename": base_filename,
            "embedding": entry.get("embedding", [])
        }
        transformed_data.append(transformed_entry)
    
    # Create the final structure for Zilliz insertion.
    zilliz_format = {
        "collectionName": "landscapes",
        "data": transformed_data
    }
    
    # Write the transformed metadata to the new JSON file.
    with open(output_file, "w") as f:
        json.dump(zilliz_format, f, indent=2)
    
    print(f"Transformed metadata saved to {output_file}")

if __name__ == "__main__":
    main()

