#!/usr/bin/env python3
"""
Doc2Vec Training Script with PV-DBOW model
Reads JSON documents, trains embeddings, and saves model and embeddings.
"""

import argparse
import json
import os
import re
from pathlib import Path
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def is_smiles(token):
    """
    Simple heuristic to detect if a token might be a SMILES string.
    SMILES typically contain characters like parentheses, brackets, numbers,
    and specific chemical symbols.
    """
    # Check if token contains typical SMILES characters
    smiles_pattern = r'[A-Z][a-z]?|\(|\)|\[|\]|=|#|@|\+|-|\\|/|%|\d'
    matches = re.findall(smiles_pattern, token)
    # If more than 50% of the token matches SMILES patterns, consider it SMILES
    return len(''.join(matches)) / len(token) > 0.5 if token else False


def preprocess_text(text):
    """
    Preprocess text: lowercase and remove punctuation, but preserve SMILES strings.
    """
    tokens = text.split()
    processed_tokens = []

    for token in tokens:
        if is_smiles(token):
            # Keep SMILES as-is
            processed_tokens.append(token)
        else:
            # Lowercase and remove punctuation for non-SMILES tokens
            cleaned = re.sub(r'[^\w\s]', '', token.lower())
            if cleaned:  # Only add non-empty tokens
                processed_tokens.append(cleaned)

    return processed_tokens


def load_documents(input_dir):
    """
    Load all JSON documents from the input directory.
    Returns a list of TaggedDocument objects and a mapping of tags to experiment_ids.
    """
    documents = []
    tag_to_experiment_id = {}

    json_files = list(Path(input_dir).glob('*.json'))
    logging.info(f"Found {len(json_files)} JSON files in {input_dir}")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if experiment_id exists
            if 'experiment_id' not in data:
                logging.warning(f"Skipping {json_file.name}: missing 'experiment_id' field")
                continue

            # Check if free_text exists
            if 'free_text' not in data:
                logging.warning(f"Skipping {json_file.name}: missing 'free_text' field")
                continue

            experiment_id = data['experiment_id']
            free_text = data['free_text']

            # Preprocess the text
            words = preprocess_text(free_text)

            if not words:
                logging.warning(f"Skipping {json_file.name}: no valid words after preprocessing")
                continue

            # Use a unique tag (index) for each document
            tag = len(documents)
            documents.append(TaggedDocument(words=words, tags=[tag]))
            tag_to_experiment_id[tag] = experiment_id

        except json.JSONDecodeError as e:
            logging.error(f"Error parsing {json_file.name}: {e}")
        except Exception as e:
            logging.error(f"Error processing {json_file.name}: {e}")

    logging.info(f"Successfully loaded {len(documents)} documents")
    return documents, tag_to_experiment_id


def train_doc2vec(documents, vector_size, min_count):
    """
    Train a Doc2Vec model using PV-DBOW.
    """
    logging.info("Building vocabulary...")

    # PV-DBOW parameters: dm=0 means DBOW mode
    model = Doc2Vec(
        documents=documents,
        vector_size=vector_size,
        min_count=min_count,
        dm=0,  # PV-DBOW (distributed bag of words)
        epochs=20,  # Sensible default
        window=5,  # Sensible default
        workers=4,  # Use multiple cores
        negative=5,  # Negative sampling
        hs=0,  # Don't use hierarchical softmax
        sample=1e-5,  # Downsampling of frequent words
        seed=42  # For reproducibility
    )

    logging.info(f"Vocabulary size: {len(model.wv)}")
    logging.info(f"Training complete. Total {len(model.dv)} document vectors.")

    return model


def save_embeddings(model, tag_to_experiment_id, output_dir):
    """
    Save embeddings to a JSON file with experiment_id as keys.
    """
    embeddings = {}

    for tag, experiment_id in tag_to_experiment_id.items():
        # Get the document vector for this tag
        vector = model.dv[tag].tolist()
        embeddings[experiment_id] = vector

    output_file = os.path.join(output_dir, 'embeddings.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, indent=2)

    logging.info(f"Embeddings saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Train Doc2Vec model on JSON documents using PV-DBOW'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing input JSON files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save the model and embeddings'
    )
    parser.add_argument(
        '--vector_size',
        type=int,
        default=100,
        help='Dimensionality of the feature vectors (default: 100)'
    )
    parser.add_argument(
        '--min_count',
        type=int,
        default=2,
        help='Ignores all words with total frequency lower than this (default: 2)'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load documents
    documents, tag_to_experiment_id = load_documents(args.input_dir)

    if not documents:
        logging.error("No valid documents found. Exiting.")
        return

    # Train model
    logging.info(f"Training Doc2Vec with vector_size={args.vector_size}, min_count={args.min_count}")
    model = train_doc2vec(documents, args.vector_size, args.min_count)

    # Save model
    model_path = os.path.join(args.output_dir, 'doc2vec_model.model')
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")

    # Save embeddings
    save_embeddings(model, tag_to_experiment_id, args.output_dir)

    logging.info("Done!")


if __name__ == '__main__':
    main()
