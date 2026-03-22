import time
import numpy as np
import faiss

dim = 384

np.random.seed(42)

num_vectors = 100000
vectors = np.random.randn(num_vectors, dim).astype(np.float32)

query = np.random.randn(1, dim).astype(np.float32)

faiss.normalize_L2(vectors)
faiss.normalize_L2(query)

# Exact search
start = time.time()
similarities = np.dot(vectors, query.T).flatten()
exact_top5 = np.argsort(similarities)[::-1][:5]
exact_time = time.time() - start

# IVF-PQ index
nlist = 100  # number of Voronoi cells
m = 8        # number of subquantizers
# https://docs.lancedb.com/indexing
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
index.train(vectors)
index.add(vectors)
index.nprobe = 10  # search more cells for better recall

start = time.time()
ann_distances, ann_indices = index.search(query, 5)
ann_time = time.time() - start

print(f"Exact Search Time: {exact_time:.5f}s")
print(f"ANN Search Time:   {ann_time:.5f}s")
print(f"Speedup:           {exact_time/ann_time:.1f}x")
