#!/usr/bin/env python3
"""
Simple client to query the Doc2Vec inference server.
"""

import socket
import json
import sys


def query_server(socket_path, text):
    """
    Send a query to the inference server and get the embedding.
    """
    # Create socket
    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    try:
        # Connect to server
        client_socket.connect(socket_path)

        # Send query
        client_socket.sendall(text.encode('utf-8'))

        # Receive response
        response = client_socket.recv(65536).decode('utf-8')
        embedding = json.loads(response)

        return embedding

    finally:
        client_socket.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python client.py 'your query text here'")
        sys.exit(1)

    socket_path = '/tmp/doc2vec_inference.sock'
    query_text = ' '.join(sys.argv[1:])

    print(f"Querying: {query_text}")
    embedding = query_server(socket_path, query_text)

    # Check for error codes
    if embedding == [-1]:
        print("Error: Model not found")
    elif embedding == [-2]:
        print("Error: Empty string")
    elif embedding == [-3]:
        print("Error: Malformed input")
    else:
        print(f"Embedding (first 10 dims): {embedding[:10]}")
        print(f"Total dimensions: {len(embedding)}")
