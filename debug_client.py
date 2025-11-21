"""Debug Qdrant client object"""
from qdrant_client import QdrantClient
import os

# Create a client like we do in the service
storage_path = "./debug_qdrant"
os.makedirs(storage_path, exist_ok=True)

try:
    client = QdrantClient(path=storage_path)
    
    print(f"Client type: {type(client)}")
    print(f"Client class: {client.__class__.__name__}")
    
    # Check if search method exists
    if hasattr(client, 'search'):
        print("✅ client.search() exists")
    else:
        print("❌ client.search() DOES NOT exist")
    
    # Check internal client
    if hasattr(client, '_client'):
        print(f"\nInternal _client type: {type(client._client)}")
        if hasattr(client._client, 'search'):
            print("✅ client._client.search() exists")
    
    # List all methods
    print("\nAll methods starting with 's':")
    for attr in dir(client):
        if attr.startswith('s') and callable(getattr(client, attr)):
            print(f"  - {attr}")
            
finally:
    # Try to clean up
    import shutil
    import time
    del client
    time.sleep(1)
    if os.path.exists(storage_path):
        try:
            shutil.rmtree(storage_path)
        except:
            pass
