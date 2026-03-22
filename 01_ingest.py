import os
import sqlite3
import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer

print("Loading sentence-transformers model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

LANCEDB_PATH = "lancedb_data"
SQLITE_PATH = "sqlite_data.db"

print("Connecting to SQLite...")
conn = sqlite3.connect(SQLITE_PATH)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS documents")
cursor.execute('''
    CREATE VIRTUAL TABLE documents USING fts5(
        filename, chunk_id UNINDEXED, text
    )
''')
conn.commit()

# Connect to LanceDB
print("Connecting to LanceDB...")
db = lancedb.connect(LANCEDB_PATH)

def process_files(data_dir="data"):
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"Directory {data_dir} is empty or does not exist.")
        return

    data_for_lancedb = []
    
    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue
            
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Simple chunking by paragraph (or double newline)
        chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
        
        for i, chunk in enumerate(chunks):
            # 1. Insert into SQLite (Full-Text Search)
            cursor.execute('''
                INSERT INTO documents (filename, chunk_id, text)
                VALUES (?, ?, ?)
            ''', (filename, i, chunk))
            
            # 2. Prepare for LanceDB (Semantic Search)
            vector = model.encode(chunk).tolist()
            data_for_lancedb.append({
                "vector": vector,
                "filename": filename,
                "chunk_id": i,
                "text": chunk
            })
            
    conn.commit()
    
    if data_for_lancedb:
        df = pd.DataFrame(data_for_lancedb)
        
        if "documents" in db.table_names():
            db.drop_table("documents")
            
        # Create table with vectors
        db.create_table("documents", data=df)
        print(f"✅ Ingested {len(data_for_lancedb)} text chunks into LanceDB and SQLite.")
    else:
        print("⚠️ No text data found to ingest.")

if __name__ == "__main__":
    process_files()
    print("Ingestion complete!")
