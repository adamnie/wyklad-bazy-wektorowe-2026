import os
import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
db = lancedb.connect("memory://")

data_dir = "data"
data = []

if os.path.exists(data_dir):
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for the specified string indicating an exam is required
            has_exam = "Forma zaliczenia\nEgzamin" in content or "Forma zaliczenia Egzamin" in content
            
            data.append({
                "filename": filename,
                "text": content,
                "exam": has_exam
            })

if not data:
    print("No data found in 'data/' directory.")
    exit()

print(f"Loaded {len(data)} documents. Checking for 'exam' attribute...")
for d in data:
    status = "Yes" if d['exam'] else "No"
    print(f" - {d['filename']}: Exam = {status}")
print()

df = pd.DataFrame(data)
# Add embeddings
df["vector"] = df["text"].apply(lambda x: model.encode(x).tolist())

tbl = db.create_table("documents", data=df)

query = "Zasady zaliczania przedmiotu"
query_vec = model.encode(query).tolist()

print(f"Query: '{query}'\n")

print("--- Search Without Filters (All Documents) ---")
res_all = tbl.search(query_vec).limit(5).to_pandas()
for _, row in res_all.iterrows():
    # Print max 100 chars
    text_preview = row['text'].replace('\n', ' ')[:80] + "..."
    print(f"[{row['filename']}] (exam={row['exam']}, dist={row['_distance']:.4f})")
    print(f"    {text_preview}")

print("\n--- Search With Metadata Filter (exam = true) ---")
# LanceDB SQL filter syntax
res_filtered = tbl.search(query_vec).where("exam = true").limit(5).to_pandas()

if res_filtered.empty:
    print("No results matched the filter.")
else:
    for _, row in res_filtered.iterrows():
        text_preview = row['text'].replace('\n', ' ')[:80] + "..."
        print(f"[{row['filename']}] (exam={row['exam']}, dist={row['_distance']:.4f})")
        print(f"    {text_preview}")
