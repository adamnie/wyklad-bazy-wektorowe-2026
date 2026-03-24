import os
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

def simple_chunker(text, max_length=200):
    """Simple chunking by paragraphs/sentences."""
    chunks = []
    for p in text.split('\n\n'):
        p = p.strip()
        if not p:
            continue
        if len(p) <= max_length:
            chunks.append(p)
        else:
            sentences = [s + '.' for s in p.split('. ') if s.strip()]
            for s in sentences:
                if len(s) > 10:
                    chunks.append(s)
    return chunks

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')

    db = lancedb.connect("memory://")

    data_dir = "data"
    all_chunks = []
    
    print(f"3. Reading text files from './{data_dir}' and chunking...")
    num_files = 0
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                num_files += 1
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_chunks = simple_chunker(content, 200)
                
                for chunk_text in file_chunks:
                    all_chunks.append({
                        "text": chunk_text, 
                        "filename": filename
                    })
                    
    print(f"   -> Processed {num_files} files into {len(all_chunks)} chunks.")
    
    if not all_chunks:
        print(f"Error: No text data found in {data_dir}")
        return

    print("4. Computing embedding vectors for all chunks...")
    data_with_embeddings = [
        {
            "text": c["text"], 
            "filename": c["filename"], 
            "vector": model.encode(c["text"]).tolist()
        } 
        for c in all_chunks
    ]
        
    print("5. Populating vector database (LanceDB)...")
    tbl = db.create_table("documents", data=pd.DataFrame(data_with_embeddings))
    
    print("-" * 60)
    
    # ==========================================================
    # Set your query string here
    # ==========================================================
    query = "How to implement simple regression in Python?"
    
    print(f"🔍 SEARCH QUERY: '{query}'")
    
    # Calculate the vector for the search query
    query_vector = model.encode(query).tolist()
    
    # Run the nearest-neighbor search (limit to 3 results)
    results_df = tbl.search(query_vector).limit(3).to_pandas()
    
    if results_df.empty:
        print("No results found.")
    else:
        for idx, row in results_df.iterrows():
            distance = row.get('_distance', 0)
            print(f"\n[{idx+1}] File: {row['filename']} (Distance: {distance:.4f})")
            print(f"    Text: {row['text'].strip()}")
            
    print("\n" + "-" * 60)

if __name__ == "__main__":
    main()
