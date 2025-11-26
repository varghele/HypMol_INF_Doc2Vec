#!/usr/bin/env python3
"""
Doc2Vec Embedding Enrichment Script
Adds embeddings and similarity information to original JSON documents.
"""

import argparse
import json
import os
import logging
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_embeddings(embeddings_path):
    """
    Load the embeddings JSON file.
    Returns a dictionary mapping experiment_id to embedding vector.
    """
    if not os.path.exists(embeddings_path):
        logging.error(f"Embeddings file not found: {embeddings_path}")
        return None

    try:
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
        logging.info(f"Loaded {len(embeddings)} embeddings from {embeddings_path}")
        return embeddings
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return None


def load_json_documents(input_dir):
    """
    Load all JSON documents from the input directory.
    Returns a dictionary mapping experiment_id to the full document data.
    """
    documents = {}
    json_files = list(Path(input_dir).glob('*.json'))

    logging.info(f"Found {len(json_files)} JSON files in {input_dir}")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'experiment_id' not in data:
                logging.warning(f"Skipping {json_file.name}: missing 'experiment_id' field")
                continue

            experiment_id = data['experiment_id']
            documents[experiment_id] = {
                'data': data,
                'filename': json_file.name
            }

        except json.JSONDecodeError as e:
            logging.error(f"Error parsing {json_file.name}: {e}")
        except Exception as e:
            logging.error(f"Error processing {json_file.name}: {e}")

    logging.info(f"Successfully loaded {len(documents)} documents")
    return documents


def calculate_similarity_matrix(embeddings):
    """
    Calculate cosine similarity matrix for all embeddings.
    Returns the similarity matrix and a list of experiment_ids in order.
    """
    experiment_ids = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[exp_id] for exp_id in experiment_ids])

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(embedding_matrix)

    logging.info(f"Calculated similarity matrix of shape {similarity_matrix.shape}")
    return similarity_matrix, experiment_ids


def find_similar_experiments(experiment_id, experiment_ids, similarity_matrix, idx):
    """
    Find similar experiments for a given experiment based on cosine similarity.
    Returns three lists: very high, high, and medium similarity experiments.
    """
    # Get similarity scores for this experiment
    similarities = similarity_matrix[idx]

    # Create list of (experiment_id, similarity) tuples, excluding self
    similar_experiments = []
    for i, other_exp_id in enumerate(experiment_ids):
        if i != idx:  # Exclude self
            similar_experiments.append({
                'experiment_id': other_exp_id,
                'similarity': float(similarities[i])
            })

    # Sort by similarity (highest first)
    similar_experiments.sort(key=lambda x: x['similarity'], reverse=True)

    # Filter into categories (non-overlapping)
    very_high = []
    high = []
    medium = []

    for exp in similar_experiments:
        sim = exp['similarity']

        if sim >= 0.95 and len(very_high) < 5:
            very_high.append(exp)
        elif sim >= 0.90 and sim < 0.95 and len(high) < 10:
            high.append(exp)
        elif sim >= 0.85 and sim < 0.90 and len(medium) < 20:
            medium.append(exp)

    return very_high, high, medium


def enrich_documents(documents, embeddings, output_dir):
    """
    Enrich documents with embeddings and similarity information.
    """
    # Check for missing embeddings
    missing_embeddings = []
    for exp_id in documents.keys():
        if exp_id not in embeddings:
            missing_embeddings.append(exp_id)
            logging.warning(f"No embedding found for experiment_id: {exp_id}")

    # Check for embeddings without documents
    missing_documents = []
    for exp_id in embeddings.keys():
        if exp_id not in documents:
            missing_documents.append(exp_id)
            logging.warning(f"No document found for experiment_id: {exp_id}")

    if missing_embeddings:
        logging.warning(f"Total documents without embeddings: {len(missing_embeddings)}")
    if missing_documents:
        logging.warning(f"Total embeddings without documents: {len(missing_documents)}")

    # Only process documents that have embeddings
    valid_exp_ids = [exp_id for exp_id in documents.keys() if exp_id in embeddings]

    if not valid_exp_ids:
        logging.error("No valid documents with embeddings found. Exiting.")
        return

    logging.info(f"Processing {len(valid_exp_ids)} documents with embeddings")

    # Create filtered embeddings dict for similarity calculation
    filtered_embeddings = {exp_id: embeddings[exp_id] for exp_id in valid_exp_ids}

    # Calculate similarity matrix
    similarity_matrix, experiment_ids = calculate_similarity_matrix(filtered_embeddings)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Enrich each document
    for idx, exp_id in enumerate(experiment_ids):
        doc_info = documents[exp_id]
        data = doc_info['data'].copy()

        # Add embedding
        data['doc2vec_embedding'] = embeddings[exp_id]

        # Find similar experiments
        very_high, high, medium = find_similar_experiments(
            exp_id, experiment_ids, similarity_matrix, idx
        )

        # Add similarity fields
        data['SIMILARITY_VERY_HIGH'] = very_high
        data['SIMILARITY_HIGH'] = high
        data['SIMILARITY_MEDIUM'] = medium

        # Save enriched document
        output_path = os.path.join(output_dir, doc_info['filename'])
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    logging.info(f"Enriched {len(experiment_ids)} documents and saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Enrich JSON documents with Doc2Vec embeddings and similarity information'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing original JSON files'
    )
    parser.add_argument(
        '--embeddings_path',
        type=str,
        required=True,
        help='Path to the embeddings JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save enriched JSON files'
    )

    args = parser.parse_args()

    # Load embeddings
    embeddings = load_embeddings(args.embeddings_path)
    if embeddings is None:
        logging.error("Failed to load embeddings. Exiting.")
        return

    # Load documents
    documents = load_json_documents(args.input_dir)
    if not documents:
        logging.error("No valid documents found. Exiting.")
        return

    # Enrich documents
    enrich_documents(documents, embeddings, args.output_dir)

    logging.info("Done!")


if __name__ == '__main__':
    main()
