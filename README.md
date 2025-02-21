# Vector Search with UltiHash

This repository contains a multimodal retrieval pipeline that combines **UltiHash** for high-performance object storage and **Zilliz** (Milvus) for vector search. It enables fast and scalable **text-to-image** retrieval using **CLIP embeddings**.

## **Overview**
Instead of relying on exact keyword matches, this pipeline allows for **semantic search**, meaning it can retrieve relevant images even if the search terms do not explicitly match filenames or metadata. The system consists of:

- **UltiHash**: Stores raw image data efficiently.
- **Zilliz (Milvus)**: Stores vector embeddings and enables similarity search.
- **CLIP**: Generates embeddings for both images and queries.
- **Flask API**: Serves as an interface for querying images using text.

---

## **1. Prerequisites**
### **Install Dependencies**
First, install the required Python packages:

```bash
pip install -r requirements.txt
```

You also need:
- **Docker** (for UltiHash, if running locally)
- A **Zilliz** cloud account (or a self-hosted Milvus instance)

---

## **2. UltiHash Setup**
UltiHash acts as the object storage backend for raw images.

1. Deploy an UltiHash cluster (local or cloud):
   - [Installation Guide](https://docs.ultihash.io/administration/3.-general-installation)

2. Create a storage bucket (e.g., `landscapes`):
   - [API Guide](https://docs.ultihash.io/development/1.-api-use)

3. Upload images to UltiHash:
```bash
uhctl upload --bucket landscapes /path/to/your/images/
```

---

## **3. Generating Image Embeddings**
Run the following script to process your images with CLIP and store their embeddings:

```bash
python generate-embeddings.py
```

This script:
- Loads images from `landscapes_test/`
- Generates embeddings using CLIP
- Saves metadata to `landscapes_metadata.json`

---

## **4. Preparing Metadata for Zilliz**
To format the metadata for Zilliz, run:

```bash
python transform-metadata.py
```

This script:
- Reads `landscapes_metadata.json`
- Formats it for Zilliz
- Saves the result as `landscape_metadata_col.json`

---

## **5. Zilliz Setup (Vector Database)**
1. Create a **Zilliz Cloud cluster**: [Zilliz Cloud](https://cloud.zilliz.com/)
2. Import your metadata collection (`landscape_metadata_col.json`).
3. Set up a collection with **COSINE similarity**.

---

## **6. Running the Retrieval API**
The Flask API enables querying images with text.

```bash
python retrieval-api.py
```

This will:
- Accept text queries (e.g., `"sunset at the beach"`)
- Convert them into embeddings using CLIP
- Perform vector search in Zilliz
- Retrieve the most relevant images from UltiHash

### **Test with cURL**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "sunset at the beach"}' http://127.0.0.1:5000/get_landscape_images
```