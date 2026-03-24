import streamlit as st
import sqlite3
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

# Page Config
st.set_page_config(page_title="Hybrid Search with RRF Demo", page_icon="🧬", layout="wide")
st.title("Hybrid Search with RRF Demo")

st.markdown("""
This application demonstrates **Reciprocal Rank Fusion (RRF)**, combining:
- **🔠 Keyword Search** (using *SQLite FTS5*)
- **🧠 Semantic Search** (using *LanceDB* and *sentence-transformers*)
""")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_lancedb_table(db_path="lancedb_data", table_name="documents"):
    try:
        db = lancedb.connect(db_path)
        if table_name in db.table_names():
            return db.open_table(table_name)
    except Exception as e:
        st.error(f"LanceDB error: {e}")
    return None

def get_sqlite_conn(db_path="sqlite_data.db"):
    return sqlite3.connect(db_path)

def compute_rrf(sqlite_results, lancedb_results, k=60):
    """
    Reciprocal Rank Fusion (RRF) combines multiple search results.
    For each document, we calculate its score using the formula:
    RRF_Score = 1 / (k + rank)
    We then sum the scores from all searches to get the final score.
    """
    # Dictionary to store combined results: { unique_id: dictionary_of_data }
    combined_results = {}
    
    # 1. Process Keyword Search Results (SQLite)
    # enumerate(..., start=1) gives us rank 1, 2, 3, etc.
    for rank, row in enumerate(sqlite_results, start=1):
        filename, chunk_id, text, _ = row
        unique_id = f"{filename}_{chunk_id}"
        
        # Initialize the record if we haven't seen this chunk yet
        if unique_id not in combined_results:
            combined_results[unique_id] = {
                "uid": unique_id,
                "text": text, 
                "filename": filename, 
                "score": 0.0
            }
            
        # Add RRF score for the keyword search rank
        combined_results[unique_id]["score"] += 1.0 / (k + rank)
        combined_results[unique_id]["sqlite_rank"] = rank
        
    # 2. Process Semantic Search Results (LanceDB)
    if lancedb_results is not None and not lancedb_results.empty:
        for i, row in lancedb_results.iterrows():
            rank = i + 1  # Make rank 1-indexed
            filename = row.get("filename", "Unknown")
            chunk_id = row.get("chunk_id", -1)
            text = row.get("text", "")
            
            unique_id = f"{filename}_{chunk_id}"
            
            # Initialize the record if we haven't seen this chunk yet
            if unique_id not in combined_results:
                combined_results[unique_id] = {
                    "uid": unique_id,
                    "text": text, 
                    "filename": filename, 
                    "score": 0.0
                }
                
            # Add RRF score for the semantic search rank
            combined_results[unique_id]["score"] += 1.0 / (k + rank)
            combined_results[unique_id]["lancedb_rank"] = rank
            
    # 3. Convert dictionary to a list and sort by final RRF score
    final_list = list(combined_results.values())
    final_list.sort(key=lambda item: item["score"], reverse=True)
    
    return final_list

with st.spinner("Loading embedding model..."):
    model = load_model()

tbl = get_lancedb_table()

st.divider()

# Search UI
col_search, col_limit = st.columns([3, 1])
with col_search:
    query = st.text_input("Enter a search query:", placeholder="e.g., 'What is machine learning?'")
with col_limit:
    limit = st.slider("Result Limit per Method", min_value=1, max_value=20, value=10)

if query:
    st.divider()
    
    # 1. Keyword search (SQLite FTS5)
    sqlite_conn = get_sqlite_conn()
    cursor = sqlite_conn.cursor()
    
    sanitized_query = query.replace('"', '""').replace("'", "''")
    query_sql = f'''
        SELECT filename, chunk_id, text, bm25(documents) as score
        FROM documents 
        WHERE documents MATCH ? 
        ORDER BY score LIMIT ?
    '''
    try:
        cursor.execute(query_sql, (sanitized_query, limit))
        sqlite_results = cursor.fetchall()
    except Exception as e:
        if "no such table" in str(e):
            st.error("SQLite data not found. Run `01_ingest.py` first.")
        else:
            st.error(f"SQLite search failed: {e}")
        sqlite_results = []
    finally:
        sqlite_conn.close()
        
    # 2. Semantic search (LanceDB)
    if tbl is not None:
        query_vector = model.encode(query).tolist()
        lancedb_results = tbl.search(query_vector).limit(limit).to_pandas()
    else:
        st.error("LanceDB search failed. Table not found. Run `01_ingest.py` first.")
        lancedb_results = None
        
    # 3. Reciprocal Rank Fusion
    final_results = compute_rrf(sqlite_results, lancedb_results, k=60)
    
    # Display Results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("🔠 Keyword Matches")
        st.caption("SQLite FTS5")
        if sqlite_results:
            for rank, row in enumerate(sqlite_results):
                filename, chunk_id, text, score = row
                with st.expander(f"#{rank+1} | 📄 {filename[:20]}", expanded=False):
                    st.write(f"**Score:** {score:.4f}")
                    st.info(text[:300] + "..." if len(text) > 300 else text)
        else:
            st.info("No explicit keyword matches found.")
            
    with col2:
        st.header("🧠 Semantic Matches")
        st.caption("LanceDB")
        if lancedb_results is not None and not lancedb_results.empty:
            for rank, row in lancedb_results.iterrows():
                dist = row.get('_distance', 0)
                filename = row.get('filename', 'Unknown')
                text = row.get('text', '')
                with st.expander(f"#{rank+1} | 📄 {filename[:20]}", expanded=False):
                    st.write(f"**Distance:** {dist:.4f}")
                    st.success(text[:300] + "..." if len(text) > 300 else text)
        else:
            st.info("No semantic matches found.")
            
    with col3:
        st.header("🧬 RRF Fused Results")
        st.caption("Combined list")
        if final_results:
            for rank, res in enumerate(final_results[:limit]):
                uid = res['uid']
                score = res['score']
                text = res['text']
                s_rank = res.get('sqlite_rank', 'N/A')
                l_rank = res.get('lancedb_rank', 'N/A')
                filename = res.get('filename', 'Unknown')
                
                with st.expander(f"🏆 Rank #{rank+1} | 📄 {filename[:20]}", expanded=True):
                    st.markdown(f"**RRF Score:** {score:.5f}")
                    st.markdown(f"**Original Ranks:** Keyword #{s_rank} | Semantic #{l_rank}")
                    st.warning(text)
        else:
            st.info("No results to fuse.")
