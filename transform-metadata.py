#!/usr/bin/env python3
import os
import json
from pathlib import Path

def main():
    input_file = Path("/users/ultihash/test/landscapes_metadata.json")
    output_file = Path("/users/ultihash/test/landscape_metadata_col.json")

    with open(input_file, "r") as f:
        metadata_list = json.load(f)

    transformed_data = []
    for entry in metadata_list:
        filename = entry.get("filename", "")
        base_filename, _ = os.path.splitext(filename)

        transformed_entry = {
            "filename": base_filename,
            "embedding": entry.get("embedding", [])
        }
        transformed_data.append(transformed_entry)

    zilliz_format = {
        "collectionName": "landscapes",
        "data": transformed_data
    }

    with open(output_file, "w") as f:
        json.dump(zilliz_format, f, indent=2)

    print(f"Transformed metadata saved to {output_file}")

if __name__ == "__main__":
    main()