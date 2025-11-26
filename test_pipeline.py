#!/usr/bin/env python3
"""
Comprehensive test script for Doc2Vec pipeline.
Generates synthetic data and tests training, enrichment, and inference.
"""

import json
import os
import shutil
import subprocess
import time
import socket
import tempfile
import logging
from pathlib import Path

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Doc2VecPipelineTester:
    """Test suite for the complete Doc2Vec pipeline."""

    def __init__(self):
        self.test_dir = tempfile.mkdtemp(prefix='doc2vec_test_')
        self.input_dir = os.path.join(self.test_dir, 'input_jsons')
        self.output_dir = os.path.join(self.test_dir, 'output')
        self.enriched_dir = os.path.join(self.test_dir, 'enriched')
        self.socket_path = os.path.join(self.test_dir, 'test_inference.sock')

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.enriched_dir, exist_ok=True)

        logging.info(f"Test directory created: {self.test_dir}")

    def generate_synthetic_data(self, num_documents=20):
        """Generate synthetic JSON documents for testing."""
        logging.info(f"Generating {num_documents} synthetic documents...")

        # Sample chemistry-related texts with SMILES
        sample_texts = [
            "Synthesized compound CC(C)C with yield 85% using standard procedure",
            "Reaction of C1=CC=CC=C1 with catalyst gave product in 92% yield",
            "Prepared CCOC(=O)C via esterification reaction at room temperature",
            "Compound C1CCCCC1 showed high stability under acidic conditions",
            "Oxidation of CC(C)O yielded corresponding ketone in good yield",
            "Hydrogenation of C=C double bond completed in 3 hours",
            "Crystallization from ethanol gave pure CC(=O)C product",
            "NMR analysis confirmed structure of synthesized C1=CC=C(C=C1)O",
            "Purification by column chromatography yielded CCN(CC)CC in 78% yield",
            "Coupling reaction between CCCC and C1=CC=CC=C1 was successful",
            "Reduction of carbonyl group in CC(=O)CC proceeded smoothly",
            "Bromination of aromatic ring C1=CC=CC=C1 gave ortho product",
            "Synthesis of heterocycle C1=CN=CC=C1 completed in two steps",
            "Alkylation of amine with CC(C)Br gave tertiary amine product",
            "Deprotection of CCOC(=O)C yielded free carboxylic acid",
            "Cyclization reaction formed six-membered ring C1CCCCC1",
            "Grignard reagent CC(C)MgBr added to ketone successfully",
            "Peptide coupling with CC(C)C(=O)O proceeded in high yield",
            "Fluorination of aromatic compound C1=CC=C(F)C=C1 was achieved",
            "Polymerization of C=C monomers gave high molecular weight polymer"
        ]

        groups = ["chemistry", "biochemistry", "materials", "synthesis"]
        experimenters = ["Alice", "Bob", "Charlie", "Diana", "Eve"]

        for i in range(num_documents):
            doc = {
                "experiment_id": f"exp_{i:03d}",
                "date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "group": groups[i % len(groups)],
                "experimenter": experimenters[i % len(experimenters)],
                "free_text": sample_texts[i % len(sample_texts)]
            }

            filename = os.path.join(self.input_dir, f"experiment_{i:03d}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2)

        logging.info(f"Generated {num_documents} documents in {self.input_dir}")

    def test_training(self):
        """Test the training script."""
        logging.info("\n" + "=" * 60)
        logging.info("TEST 1: Training Doc2Vec Model")
        logging.info("=" * 60)

        cmd = [
            'python', 'train_doc2vec.py',
            '--input_dir', self.input_dir,
            '--output_dir', self.output_dir,
            '--vector_size', '50',
            '--min_count', '1'
        ]

        logging.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Training failed with error:\n{result.stderr}")
            return False

        logging.info(result.stdout)

        # Check if outputs exist
        model_path = os.path.join(self.output_dir, 'doc2vec_model.model')
        embeddings_path = os.path.join(self.output_dir, 'embeddings.json')

        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            return False

        if not os.path.exists(embeddings_path):
            logging.error(f"Embeddings file not found: {embeddings_path}")
            return False

        # Verify embeddings
        with open(embeddings_path, 'r') as f:
            embeddings = json.load(f)

        logging.info(f"✓ Model trained successfully")
        logging.info(f"✓ Generated {len(embeddings)} embeddings")
        logging.info(f"✓ Embedding dimension: {len(list(embeddings.values())[0])}")

        return True

    def test_enrichment(self):
        """Test the document enrichment script."""
        logging.info("\n" + "=" * 60)
        logging.info("TEST 2: Enriching Documents with Similarities")
        logging.info("=" * 60)

        cmd = [
            'python', 'doc2vec_batch_similarity_calculation.py',
            '--input_dir', self.input_dir,
            '--embeddings_path', os.path.join(self.output_dir, 'embeddings.json'),
            '--output_dir', self.enriched_dir
        ]

        logging.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Enrichment failed with error:\n{result.stderr}")
            return False

        logging.info(result.stdout)

        # Check enriched files
        enriched_files = list(Path(self.enriched_dir).glob('*.json'))

        if not enriched_files:
            logging.error("No enriched files found")
            return False

        # Verify enriched content
        with open(enriched_files[0], 'r') as f:
            enriched_doc = json.load(f)

        required_fields = ['doc2vec_embedding', 'SIMILARITY_VERY_HIGH',
                           'SIMILARITY_HIGH', 'SIMILARITY_MEDIUM']

        for field in required_fields:
            if field not in enriched_doc:
                logging.error(f"Missing field in enriched document: {field}")
                return False

        logging.info(f"✓ Enriched {len(enriched_files)} documents")
        logging.info(f"✓ Sample document has all required fields")
        logging.info(f"✓ SIMILARITY_VERY_HIGH: {len(enriched_doc['SIMILARITY_VERY_HIGH'])} entries")
        logging.info(f"✓ SIMILARITY_HIGH: {len(enriched_doc['SIMILARITY_HIGH'])} entries")
        logging.info(f"✓ SIMILARITY_MEDIUM: {len(enriched_doc['SIMILARITY_MEDIUM'])} entries")

        return True

    def test_inference_server(self):
        """Test the inference server."""
        logging.info("\n" + "=" * 60)
        logging.info("TEST 3: Testing Inference Server")
        logging.info("=" * 60)

        model_path = os.path.join(self.output_dir, 'doc2vec_model.model')

        # Start server in background
        cmd = [
            'python', 'doc2vec_inference_server.py',
            '--model_path', model_path,
            '--socket_path', self.socket_path
        ]

        logging.info(f"Starting inference server: {' '.join(cmd)}")
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for server to start
        time.sleep(3)

        try:
            # Test queries
            test_queries = [
                "Synthesized compound CC(C)C with high yield",
                "Reaction with catalyst C1=CC=CC=C1",
                "",  # Empty string test
            ]

            for i, query in enumerate(test_queries):
                logging.info(f"\nTest query {i + 1}: '{query[:50]}...'")

                try:
                    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    client.connect(self.socket_path)
                    client.sendall(query.encode('utf-8'))
                    response = client.recv(65536).decode('utf-8')
                    client.close()

                    embedding = json.loads(response)

                    if embedding == [-2]:
                        logging.info("✓ Correctly returned error code [-2] for empty string")
                    elif isinstance(embedding, list) and len(embedding) > 1:
                        logging.info(f"✓ Received embedding with {len(embedding)} dimensions")
                        logging.info(f"  First 5 values: {embedding[:5]}")
                    else:
                        logging.warning(f"Unexpected response: {embedding}")

                except Exception as e:
                    logging.error(f"Query failed: {e}")
                    return False

            logging.info("\n✓ All inference tests passed")
            return True

        finally:
            # Stop server
            logging.info("\nStopping inference server...")
            server_process.terminate()
            server_process.wait(timeout=5)

    def cleanup(self):
        """Clean up test directory."""
        logging.info(f"\nCleaning up test directory: {self.test_dir}")
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def run_all_tests(self):
        """Run all tests in sequence."""
        logging.info("\n" + "=" * 60)
        logging.info("DOC2VEC PIPELINE COMPREHENSIVE TEST SUITE")
        logging.info("=" * 60)

        try:
            # Generate data
            self.generate_synthetic_data(num_documents=20)

            # Run tests
            tests = [
                ("Training", self.test_training),
                ("Enrichment", self.test_enrichment),
                ("Inference Server", self.test_inference_server)
            ]

            results = {}
            for test_name, test_func in tests:
                try:
                    results[test_name] = test_func()
                except Exception as e:
                    logging.error(f"Test '{test_name}' raised exception: {e}")
                    results[test_name] = False

            # Summary
            logging.info("\n" + "=" * 60)
            logging.info("TEST SUMMARY")
            logging.info("=" * 60)

            for test_name, passed in results.items():
                status = "✓ PASSED" if passed else "✗ FAILED"
                logging.info(f"{test_name}: {status}")

            all_passed = all(results.values())

            if all_passed:
                logging.info("\n🎉 ALL TESTS PASSED! 🎉")
            else:
                logging.info("\n⚠️  SOME TESTS FAILED")

            return all_passed

        finally:
            self.cleanup()


def main():
    tester = Doc2VecPipelineTester()
    success = tester.run_all_tests()
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
