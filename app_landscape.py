#!/usr/bin/env python3
import sys
from pathlib import Path

# Add the directory containing uh_download.py to sys.path.
sys.path.insert(0, "/Users/ultihash/test")

import os
import base64
from flask import Flask, request, jsonify
from pymilvus import connections, Collection
import boto3
from io import BytesIO
from PIL import Image
import argparse
import torch
from transformers import CLIPProcessor, CLIPModel

# Import the downloader class from your uh_download.py script.
from uh_download import downloader

# ---------------------------
# UltiHash License Key
# ---------------------------
# Replace the placeholder below with your actual UltiHash license key.
uh_license = "mem_cm6hrjwu703uf0sj12rcqfdpb:10240:pVeRPm/dehfGqVgNdDaPI8w85QEwYyPiYLOYQgAu7HcBwQo679haoWpzw1tLRFbSa8PP6R4JT0ePW06ggbnwCw=="  # This is the placeholder for your UltiHash license key.

# ---------------------------
# Connect to Zilliz (Milvus)
# ---------------------------
connections.connect(
    alias="default",
    uri="https://in03-d55be40bce35bc6.serverless.gcp-us-west1.cloud.zilliz.com",     # Replace with your Zilliz URI
    token="29e784a922a796a54ed6423340c8525ed84d26dfd48a85dad779279fac7f127ffcb32c1c9cfcc27c63a47f13943088cde5b1baf0"  # Replace with your Zilliz token
)
print("Connected successfully to Milvus!")
collection_name = "landscapes"  # Assuming your Zilliz collection for landscapes is named 'landscapes'
collection = Collection(collection_name)



# ---------------------------
# Set up boto3 S3 Client for UltiHash
# ---------------------------
s3 = boto3.client(
    's3',
    endpoint_url="http://127.0.0.1:8080",  # Adjust if necessary
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
)

def add_license_header(request, **kwargs):
    request.headers["UH_LICENSE_STRING"] = uh_license
        
s3.meta.events.register("before-sign.s3", add_license_header)
bucket = "landscapes"  # Set the bucket name to your landscapes bucket

# ---------------------------
# Load CLIP text encoder (for query vector generation)
# ---------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---------------------------
# Helper function: Query Zilliz with vector search
# ---------------------------
def query_landscape(query_text: str, top_k: int = 3):
    """
    Computes the CLIP text embedding for the given query and performs a vector search in Zilliz.
    
    Parameters:
      query_text (str): The text query (e.g. "mountain").
      top_k (int): The number of top results to retrieve (default 3).
    
    Returns:
      list: A list of filenames (without extension) from the best matching records.
    """
    # Compute the CLIP text embedding for the query text.
    inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_vector = clip_model.get_text_features(**inputs)[0].tolist()
    
    # Define search parameters for cosine similarity.
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    
    try:
        search_results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            # No additional filter here since we're doing a pure vector search.
            output_fields=["filename"]
        )
    except Exception as e:
        print(f"Vector search error for query '{query_text}': {e}")
        return []
    
    filenames = []
    if search_results and len(search_results) > 0 and len(search_results[0]) > 0:
        for hit in search_results[0]:
            # Access the underlying entity. Depending on your pymilvus version, you might need to use attributes.
            filenames.append(hit.filename)  # Assuming hit.filename holds the stored filename (without extension)
            print(f"🔍 Retrieved filename from Zilliz: {hit.filename}")
    else:
        print(f"No results found for query '{query_text}'.")
    return filenames

# ---------------------------
# Initialize Flask Application
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Endpoint: /get_landscape_images
# ---------------------------
@app.route("/get_landscape_images", methods=["POST"])
def get_landscape_images():
    """
    Expects a JSON payload:
    {
      "query": "mountain"
    }
    Performs a vector search in Zilliz for the given query text, retrieves the top 3 matching records,
    fetches the corresponding images from UltiHash (bucket: landscapes) using the filename with '.jpg',
    and returns the results with base64-encoded images.
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request."}), 400

    query_text = data["query"].strip()
    results = []
    target_dir = Path("/Users/ultihash/test/retrieval-test")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Get top 3 matching filenames from Zilliz.
    filenames = query_landscape(query_text, top_k=3)
    if not filenames:
        return jsonify({"error": "No matching records found."}), 404

    for filename in filenames:
        # Append .jpg to build the expected key in UltiHash.
        file_key = f"{filename}.jpg"
        try:
            response = s3.get_object(Bucket=bucket, Key=file_key)
            file_data = response["Body"].read()
        except Exception as e:
            results.append({"filename": filename, "error": f"Failed to fetch image '{file_key}': {str(e)}"})
            continue

        try:
            # Open the image using Pillow directly from memory.
            image = Image.open(BytesIO(file_data))
            image.show()  # This will open the image using your default image viewer.
        except Exception as e:
            results.append({"filename": filename, "error": f"Failed to open image: {str(e)}"})
            continue

        results.append({
            "filename": filename,
            "message": "Image fetched and opened successfully."
        })
        print(f"Processed filename {filename} and opened image {file_key}")
            
    return jsonify({"results": results})
     
# ---------------------------
# Run the Flask Application
# ---------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

