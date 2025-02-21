#!/usr/bin/env python3
import os
import json
import torch
import boto3
from pathlib import Path
from flask import Flask, request, jsonify
from pymilvus import connections, Collection
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Connect to Zilliz
connections.connect(
    alias="default",
    uri="https://<your-zilliz-uri>",
    token="<your-zilliz-token>"
)
print("✅ Connected to Zilliz!")

collection = Collection("landscapes")

# Setup boto3 for UltiHash
s3 = boto3.client(
    's3',
    endpoint_url="http://127.0.0.1:8080",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
)

bucket = "landscapes"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def query_landscape(query_text: str, top_k: int = 3):
    inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_vector = clip_model.get_text_features(**inputs)[0].tolist()

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

    try:
        search_results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["filename"]
        )
    except Exception as e:
        print(f"Vector search error: {e}")
        return []

    filenames = [hit.filename for hit in search_results[0]] if search_results else []
    return filenames

@app.route("/get_landscape_images", methods=["POST"])
def get_landscape_images():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request."}), 400

    query_text = data["query"].strip()
    filenames = query_landscape(query_text, top_k=3)
    if not filenames:
        return jsonify({"error": "No matching records found."}), 404

    results = []
    for filename in filenames:
        file_key = f"{filename}.jpg"
        try:
            response = s3.get_object(Bucket=bucket, Key=file_key)
            file_data = response["Body"].read()
            image = Image.open(BytesIO(file_data))
            image.show()
        except Exception as e:
            results.append({"filename": filename, "error": str(e)})
            continue

        results.append({"filename": filename, "message": "Image retrieved successfully."})
        print(f"✅ Retrieved {filename}")

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)