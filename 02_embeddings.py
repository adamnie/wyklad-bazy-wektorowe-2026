from sentence_transformers import SentenceTransformer

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
model = SentenceTransformer('all-MiniLM-L6-v2')
text = "Bazy wektorowe"
vector = model.encode(text)

print(f"Text: '{text}'")
print(f"Vector Dimensions: {len(vector)}")
print(f"First 10 values: {vector[:10]}")
print(f"Max: {vector.max()}")
print(f"Min: {vector.min()}")
