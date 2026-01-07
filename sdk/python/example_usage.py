from merkledb import MerkleDb
import numpy as np

# 1. Connect to MerkleDb
client = MerkleDb(url="http://localhost:4000", api_token="ABC-123")

collection = "python_sdk_test"

# 2. Setup collection
try:
    print(f"Creating collection '{collection}'...")
    client.create_collection(collection, dim=128, precision="f32")
except Exception as e:
    print(f"Collection might already exist: {e}")

# 3. Insert data using Numpy
print("Inserting numpy vectors...")
# Create 10 random vectors of dim 128
data = []
for i in range(10):
    vec = np.random.rand(128).astype('float32')
    data.append({
        "id": f"numpy_vec_{i}",
        "vector": vec,
        "metadata": {"source": "numpy", "index": i}
    })

client.insert(collection, data)

# 4. Insert data using Text (Semantic)
print("Inserting semantic text...")
semantic_data = [
    {"id": "text_1", "text": "Deep learning is a subset of machine learning.", "metadata": {"topic": "AI"}},
    {"id": "text_2", "text": "The Bible is a collection of religious texts.", "metadata": {"topic": "Religion"}}
]
client.insert(collection, semantic_data)

# 5. Search using a Vector
print("\nSearching by vector...")
query_vec = np.random.rand(128).astype('float32')
results = client.search(collection, vector=query_vec, k=3)
for res in results:
    print(f" Hit: {res['id']}, Score: {res['score']:.4f}")

# 6. Search using Text (Semantic Search)
print("\nSearching by text (Semantic)...")
results = client.search(collection, text="Tell me about neural networks", k=2)
for res in results:
    print(f" Hit: {res['id']}, Score: {res['score']:.4f}")

# 7. Cleanup
print(f"\nDropping collection '{collection}'...")
# client.drop_collection(collection)
print("Done!")
