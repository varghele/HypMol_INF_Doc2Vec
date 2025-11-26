#!/usr/bin/env python3
"""
Doc2Vec Inference Server using Unix Domain Socket
Loads a trained Doc2Vec model and serves inference requests via Unix socket.
"""

import argparse
import json
import os
import re
import socket
import logging
from pathlib import Path
from gensim.models.doc2vec import Doc2Vec

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Error codes
ERROR_MODEL_NOT_FOUND = [-1]
ERROR_EMPTY_STRING = [-2]
ERROR_MALFORMED_INPUT = [-3]


def is_smiles(token):
    """
    Simple heuristic to detect if a token might be a SMILES string.
    SMILES typically contain characters like parentheses, brackets, numbers,
    and specific chemical symbols.
    """
    smiles_pattern = r'[A-Z][a-z]?|\(|\)|\[|\]|=|#|@|\+|-|\\|/|%|\d'
    matches = re.findall(smiles_pattern, token)
    return len(''.join(matches)) / len(token) > 0.5 if token else False


def preprocess_text(text):
    """
    Preprocess text: lowercase and remove punctuation, but preserve SMILES strings.
    This must match the preprocessing used during training.
    """
    tokens = text.split()
    processed_tokens = []

    for token in tokens:
        if is_smiles(token):
            processed_tokens.append(token)
        else:
            cleaned = re.sub(r'[^\w\s]', '', token.lower())
            if cleaned:
                processed_tokens.append(cleaned)

    return processed_tokens


def load_model(model_path):
    """
    Load the trained Doc2Vec model.
    Returns the model or None if not found.
    """
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return None

    try:
        model = Doc2Vec.load(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        logging.info(f"Vector size: {model.vector_size}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None


def infer_vector(model, text):
    """
    Infer a document vector for the given text.
    Returns the embedding as a list or an error code.
    """
    # Check for empty string
    if not text or not text.strip():
        logging.warning("Received empty string")
        return ERROR_EMPTY_STRING

    # Check if model is loaded
    if model is None:
        logging.warning("Model not loaded, returning error code")
        return ERROR_MODEL_NOT_FOUND

    try:
        # Preprocess the text
        words = preprocess_text(text)

        if not words:
            logging.warning("No valid words after preprocessing")
            return ERROR_EMPTY_STRING

        # Infer vector
        vector = model.infer_vector(words)
        return vector.tolist()

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        return ERROR_MALFORMED_INPUT


def start_server(socket_path, model_path):
    """
    Start the Unix domain socket server.
    """
    # Load the model
    model = load_model(model_path)
    if model is None:
        logging.warning("Starting server without loaded model (will return error codes)")

    # Remove socket file if it already exists
    if os.path.exists(socket_path):
        os.remove(socket_path)
        logging.info(f"Removed existing socket file: {socket_path}")

    # Create Unix domain socket
    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind(socket_path)
    server_socket.listen(1)

    logging.info(f"Server listening on Unix socket: {socket_path}")
    logging.info("Press Ctrl+C to stop the server")

    try:
        while True:
            # Accept connection
            conn, _ = server_socket.accept()

            try:
                # Receive data (up to 64KB)
                data = conn.recv(65536).decode('utf-8')

                if not data:
                    continue

                logging.info(f"Received query: {data[:100]}...")  # Log first 100 chars

                # Infer vector
                result = infer_vector(model, data)

                # Send response as JSON
                response = json.dumps(result)
                conn.sendall(response.encode('utf-8'))

                logging.info(f"Sent response with {len(result)} dimensions")

            except Exception as e:
                logging.error(f"Error handling request: {e}")
                error_response = json.dumps(ERROR_MALFORMED_INPUT)
                try:
                    conn.sendall(error_response.encode('utf-8'))
                except:
                    pass

            finally:
                conn.close()

    except KeyboardInterrupt:
        logging.info("\nShutting down server...")

    finally:
        server_socket.close()
        if os.path.exists(socket_path):
            os.remove(socket_path)
            logging.info(f"Removed socket file: {socket_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Doc2Vec Inference Server using Unix Domain Socket'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained Doc2Vec model file'
    )
    parser.add_argument(
        '--socket_path',
        type=str,
        default='/tmp/doc2vec_inference.sock',
        help='Path to the Unix domain socket file (default: /tmp/doc2vec_inference.sock)'
    )

    args = parser.parse_args()

    start_server(args.socket_path, args.model_path)


if __name__ == '__main__':
    main()
