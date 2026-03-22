import streamlit as st
import sqlite3
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

# Page Config
st.set_page_config(page_title="Vector vs Text Search Demo", page_icon="🔍", layout="wide")

st.title("Vector vs Text Search Demo")
st.markdown("""
This application compares **Keyword Search** (using *SQLite FTS5*) with **Semantic Search** (using *LanceDB* and *sentence-transformers*).
""")

# Setup Models and Database Connections
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_lancedb_table():
    try:
        db = lancedb.connect("lancedb_data")
        if "documents" in db.table_names():
            return db.open_table("documents")
    except Exception as e:
        print(f"LanceDB error: {e}")
    return None

def get_sqlite_conn():
    return sqlite3.connect("sqlite_data.db")

# Load resources
with st.spinner("Loading embedding model..."):
    model = load_model()

tbl = get_lancedb_table()

# Top Search Bar
query = st.text_input("Enter your search query:", placeholder="e.g., machine learning algorithms", help="Type a sentence or keywords.")

if query:
    st.divider()
    
    # 2 side-by-side columns
    col1, col2 = st.columns(2)
    
    # --- Left Column: SQLite ---
    with col1:
        with st.expander("🔠 Keyword Search (SQLite FTS5)", expanded=False):
            try:
                conn = get_sqlite_conn()
                cursor = conn.cursor()
                
                # Sanitize input slightly to avoid FTS5 syntax errors with quotes
                sanitized_query = query.replace('"', '""').replace("'", "''")
                
                # In SQLite FTS5, bm25() score is typically more negative for better matches.
                query_sql = f'''
                    SELECT filename, text, bm25(documents) as score
                    FROM documents 
                    WHERE documents MATCH ? 
                    ORDER BY score LIMIT 5
                '''
                
                cursor.execute(query_sql, (sanitized_query,))
                sqlite_results = cursor.fetchall()
                
                if sqlite_results:
                    for row in sqlite_results:
                        filename, text, score = row
                        
                        with st.container(border=True):
                            st.markdown(f"**📄 {filename}** *(BM25 Score: {score:.4f})*")
                            st.write(text)
                else:
                    st.info("No exact keyword matches found.")
                    
                conn.close()
            except sqlite3.OperationalError as e:
                if "no such table" in str(e):
                    st.warning("SQLite data not found. Please run `python 01_ingest.py` first.")
                elif "syntax error" in str(e):
                    st.info("Invalid search syntax for FTS5. Try removing special characters.")
                else:
                    st.error(f"SQLite FTS5 Error: {e}")
            except Exception as e:
                st.error(f"SQLite Error: {e}")

    # --- Right Column: LanceDB ---
    with col2:
        with st.expander("🧠 Semantic Search (LanceDB)", expanded=False):
            if tbl is not None:
                # Generate embedding for the query
                query_vector = model.encode(query).tolist()
                
                # The distance metric used by default is L2. 
                # We can convert distance to a "Similarity" score for better UX, or just show distance.
                try:
                    results = tbl.search(query_vector).limit(5).to_pandas()
                    
                    if not results.empty:
                        for idx, row in results.iterrows():
                            # Lower distance = closer match.
                            dist = row.get('_distance', 0)
                            file_name = row.get('filename', 'Unknown')
                            text = row.get('text', '')
                            
                            with st.container(border=True):
                                st.markdown(f"**📄 {file_name}** *(Distance: {dist:.4f})*")
                                st.write(text)
                    else:
                        st.info("No semantic matches found.")
                except Exception as e:
                    st.error(f"LanceDB Query Error: {e}")
            else:
                st.warning("LanceDB data not found. Please run `python 01_ingest.py` first.")
