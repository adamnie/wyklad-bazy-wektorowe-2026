from sentence_transformers import SentenceTransformer, util

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
model = SentenceTransformer('all-MiniLM-L6-v2')

reference_text = "Lecture on vector database and semantic search"
text_A = "Textual algorithms"
text_B = "Object oriented programming concepts"

reference = model.encode(reference_text)
embedding_A = model.encode(text_A)
embedding_B = model.encode(text_B)

sim_A = util.cos_sim(reference, embedding_A).item()
sim_B = util.cos_sim(reference, embedding_B).item()

print(f"Similarity between `{reference_text}` and `{text_A}`: {sim_A:.4f}")
print(f"Similarity between `{reference_text}` and `{text_B}`: {sim_B:.4f}")

