import sqlite3
import argparse
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

def get_sqlite_conn(db_path="sqlite_data.db"):
    return sqlite3.connect(db_path)

def get_lancedb_table(db_path="lancedb_data", table_name="documents"):
    try:
        db = lancedb.connect(db_path)
        if table_name in db.table_names():
            return db.open_table(table_name)
    except Exception as e:
        print(f"LanceDB error: {e}")
    return None

def compute_rrf(sqlite_results, lancedb_results, k=60):
    # RRF Score = 1 / (k + rank)
    rrf_scores = {}
    
    # Process SQLite results (assuming they are primarily text matches)
    for rank, row in enumerate(sqlite_results):
        filename, chunk_id, text, score = row
        # create a unique key
        uid = f"{filename}_{chunk_id}"
        rrf_scores[uid] = rrf_scores.get(uid, {"text": text, "score": 0.0})
        rrf_scores[uid]["score"] += 1.0 / (k + rank + 1)
        rrf_scores[uid]["sqlite_rank"] = rank + 1
        
    # Process LanceDB results
    if lancedb_results is not None and not lancedb_results.empty:
        for rank, row in lancedb_results.iterrows():
            filename = row.get("filename", "Unknown")
            chunk_id = row.get("chunk_id", -1)
            text = row.get("text", "")
            
            uid = f"{filename}_{chunk_id}"
            rrf_scores[uid] = rrf_scores.get(uid, {"text": text, "score": 0.0})
            rrf_scores[uid]["score"] += 1.0 / (k + rank + 1)
            rrf_scores[uid]["lancedb_rank"] = rank + 1
            
    # Convert back to list and sort by RRF score descending
    final_list = []
    for uid, data in rrf_scores.items():
        data["uid"] = uid
        final_list.append(data)
        
    final_list.sort(key=lambda x: x["score"], reverse=True)
    return final_list

def main():
    parser = argparse.ArgumentParser(description="Hybrid Search with RRF Demo")
    parser.add_argument("query", type=str, nargs="?", default="What is machine learning?", help="Search query")
    parser.add_argument("--limit", type=int, default=10, help="Number of results to retrieve per method")
    args = parser.parse_args()
    
    print(f"Loading embedding model for semantic search...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    print(f"Executing Query: '{args.query}'\n")
    
    # 1. Keyword search (SQLite FTS5)
    sqlite_conn = get_sqlite_conn()
    cursor = sqlite_conn.cursor()
    
    sanitized_query = args.query.replace('"', '""').replace("'", "''")
    query_sql = f'''
        SELECT filename, chunk_id, text, bm25(documents) as score
        FROM documents 
        WHERE documents MATCH ? 
        ORDER BY score LIMIT ?
    '''
    try:
        cursor.execute(query_sql, (sanitized_query, args.limit))
        sqlite_results = cursor.fetchall()
    except Exception as e:
        print(f"SQLite search failed (did you run 01_ingest.py?): {e}")
        sqlite_results = []
    finally:
        sqlite_conn.close()
        
    # 2. Semantic search (LanceDB)
    tbl = get_lancedb_table()
    if tbl is not None:
        query_vector = model.encode(args.query).tolist()
        lancedb_results = tbl.search(query_vector).limit(args.limit).to_pandas()
    else:
        print("LanceDB search failed. Table not found.")
        lancedb_results = None
        
    # 3. Reciprocal Rank Fusion
    final_results = compute_rrf(sqlite_results, lancedb_results, k=60)
    
    # 4. output
    print("=====================================================================")
    print(" HYBRID SEARCH RESULTS (Combines Keyword + Vector, Sorted by RRF)  ")
    print("=====================================================================")
    
    if not final_results:
        print("No results found.")
        return
        
    for i, res in enumerate(final_results[:args.limit]):
        uid = res['uid']
        score = res['score']
        text = res['text']
        s_rank = res.get('sqlite_rank', 'N/A')
        l_rank = res.get('lancedb_rank', 'N/A')
        
        # trim text for console display
        display_text = text.replace('\n', ' ')
        if len(display_text) > 80:
            display_text = display_text[:77] + "..."
            
        print(f"[{i+1}] UID: {uid:<20} | RRF Score: {score:.5f}")
        print(f"    Keyword Rank: {s_rank:<5} | Semantic Rank: {l_rank:<5}")
        print(f"    Text: {display_text}")
        print("-" * 69)

if __name__ == "__main__":
    main()
