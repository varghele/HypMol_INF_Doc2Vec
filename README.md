# HypMol_INF_Doc2Vec
Doc2Vec embedding pipeline for HypMol's Knowledge Graph. This repository provides scripts for training Doc2Vec models, running inference via a Unix domain socket server, and enriching documents with embeddings and similarity metrics.
## Table of Contents

- [Setup](#setup)
- [Scripts Overview](#scripts-overview)
- [Usage](#usage)
  - [1. Training the Model](#1-training-the-model)
  - [2. Running the Inference Server](#2-running-the-inference-server)
  - [3. Enriching Documents](#3-enriching-documents)
- [Input/Output Formats](#inputoutput-formats)
- [Features](#features)
## Setup
### Prerequisites
- Python 3.9 or higher
- Conda (recommended) or pip
### Installation
**Create and activate a new conda environment:**
```bash
conda create -n hypmol_doc2vec python=3.12
conda activate hypmol_doc2vec
```
**Install dependencies:**
```bash
pip install -r requirements.txt
```
**requirements.txt:**
```text
gensim==4.3.2
scikit-learn==1.3.2
```
---
## Scripts Overview
1. `train_doc2vec.py` \
Trains a Doc2Vec model using the PV-DBOW algorithm on a directory of JSON documents.\
**Features:**
   - Reads JSON files with experiment_id and free_text fields
   - Preprocesses text (lowercase, punctuation removal) while preserving SMILES strings
   - Automatically builds vocabulary and trains the model
   - Saves trained model as doc2vec_model.model
   - Exports embeddings to embeddings.json with experiment IDs as keys


2. `doc2vec_inference_server.py` \
Runs a background inference server using Unix domain sockets for real-time embedding generation.\
**Features:**
   - Loads the trained model once and keeps it in memory
   - Accepts text queries via Unix socket
   - Returns embedding vectors as JSON arrays
   - Uses the same preprocessing as training
   - Error codes: `[-1]` (model not found), `[-2]`(empty string),`[-3]` (malformed input)


3. `doc2vec_batch_similarity_calculation.py`\
Enriches original JSON documents with embeddings and similarity information (designed for nightly batch processing).\
**Features:**
   - Adds `doc2vec_embedding` field to each document
   - Calculates cosine similarity between all documents
   - Adds three similarity fields with top-k similar experiments:
     - `SIMILARITY_VERY_HIGH`: up to 5 experiments with similarity ≥ 0.95
     - `SIMILARITY_HIGH`: up to 10 experiments with similarity ≥ 0.90
     - `SIMILARITY_MEDIUM`: up to 20 experiments with similarity ≥ 0.85
   - Non-overlapping categories (experiments appear only in their highest category)

## Usage
1. **Training the Model**\
Train a Doc2Vec model on your JSON documents:
```bash
python train_doc2vec.py \
    --input_dir /path/to/json/files \
    --output_dir /path/to/output \
    --vector_size 128 \
    --min_count 2
```

**Arguments:**
- `--input_dir`: Directory containing input JSON files
- `--output_dir`: Directory to save the model and embeddings
- `--vector_size`: Dimensionality of feature vectors (default: 100)
- `--min_count`: Minimum word frequency threshold (default: 2)

**Output:** 
- `doc2vec_model.model`: Trained Doc2Vec model
- `embeddings.json`: Document embeddings mapped by experiment_id
---
2. **Running the Inference Server** \
Start the background inference server for real-time queries:
```bash
python doc2vec_inference_server.py \
    --model_path ./output/doc2vec_model.model \
    --socket_path /tmp/doc2vec_inference.sock
```

**Arguments:**
- `--model_path`: Path to the trained model file
- `--socket_path`: Path to Unix domain socket (default: /tmp/doc2vec_inference.sock)

**Querying the server:**\
Use the provided client script:
```bash
python client.py "This is my search query with SMILES CC(C)C"
```
Or integrate into your Python code:
```python
import socket
import json

def get_embedding(text, socket_path='/tmp/doc2vec_inference.sock'):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(socket_path)
    client.sendall(text.encode('utf-8'))
    response = client.recv(65536).decode('utf-8')
    client.close()
    return json.loads(response)

# Use it
embedding = get_embedding("my search query")
```

**Stopping the server:**\
Press `Ctrl+C` or kill the process.
---
3. **Enriching Documents**\
Enrich JSON documents with embeddings and similarity information (run nightly via cron or scheduler):
```bash
python doc2vec_batch_similarity_calculation.py \
    --input_dir ./data/jsons \
    --embeddings_path ./output/embeddings.json \
    --output_dir ./enriched_output
```

**Arguments:**
- `--input_dir`: Directory containing original JSON files
- `--embeddings_path`: Path to the embeddings JSON file
- `--output_dir`: Directory to save enriched JSON files

## Input/Output Formats
### Input JSON Structure
```json
{
  "experiment_id": "exp_001",
  "date": "2024-01-15",
  "group": "chemistry",
  "experimenter": "John Doe",
  "free_text": "Synthesized compound CC(C)C with yield 85%"
}

```
### Embeddings JSON Structure
```json
{
  "exp_001": [0.123, -0.456, 0.789, ...],
  "exp_002": [0.234, 0.567, -0.123, ...]
}
```

### Enriched Output JSON Structure
```json
{
  "experiment_id": "exp_001",
  "date": "2024-01-15",
  "group": "chemistry",
  "experimenter": "John Doe",
  "free_text": "Synthesized compound CC(C)C with yield 85%",
  "doc2vec_embedding": [0.123, -0.456, 0.789, ...],
  "SIMILARITY_VERY_HIGH": [
    {"experiment_id": "exp_042", "similarity": 0.97},
    {"experiment_id": "exp_103", "similarity": 0.96}
  ],
  "SIMILARITY_HIGH": [
    {"experiment_id": "exp_055", "similarity": 0.93},
    {"experiment_id": "exp_088", "similarity": 0.91}
  ],
  "SIMILARITY_MEDIUM": [
    {"experiment_id": "exp_022", "similarity": 0.88},
    {"experiment_id": "exp_067", "similarity": 0.86}
  ]
}
```
## Features
- **SMILES Preservation**: Automatically detects and preserves SMILES strings during text preprocessing
- **PV-DBOW Model**: Uses distributed bag of words for efficient training
- **Fast Inference**: Unix domain socket server keeps model in memory for quick queries
- **Similarity Analysis**: Automatically calculates and categorizes similar experiments
- **Scalable**: Designed for large document collections with efficient batch processing
- **Error Handling**: Comprehensive logging and error codes for debugging