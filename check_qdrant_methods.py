"""Quick test to check Qdrant client methods"""
from qdrant_client import QdrantClient

client = QdrantClient(path="./test_qdrant")

print("Available search methods:")
methods = [m for m in dir(client) if 'search' in m.lower() or 'query' in m.lower()]
for method in methods:
    print(f"  - {method}")

# Clean up
import shutil
import os
if os.path.exists("./test_qdrant"):
    shutil.rmtree("./test_qdrant")
