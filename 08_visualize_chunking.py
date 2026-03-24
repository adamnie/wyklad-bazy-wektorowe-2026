import os
import streamlit as st
import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer

# Page Config
st.set_page_config(page_title="Chunking Strategies Demo", page_icon="✂️", layout="wide")
st.title("Chunking Strategies Demo")

st.markdown("""
This application demonstrates three different text chunking strategies using documents from the `data/` directory. All sizes are tuned to output chunks of roughly **250 characters** for a fair comparison:
- **🔲 Fixed Size**: Strict character-limit splits (exactly 250 chars).
- **🌳 Recursive**: Meaning-preserving splits (falling back from paragraphs to sentences, up to 250 chars).
- **🪟 Sliding Window**: Overlapping sentences (chunks of ~3 sentences, overlap of 1) to show context shifts.
""")

def chunk_fixed_size(text, chunk_size=120):
    clean_text = text.replace('\n', ' ')
    return [clean_text[i:i+chunk_size] for i in range(0, len(clean_text), chunk_size)]

def chunk_sliding_window(text, window_size=3, overlap=1):
    # Split text into sentences using simple heuristics
    sentences = [s.strip() + '.' for s in text.replace('\n', ' ').split('. ') if s.strip()]
    
    chunks = []
    if len(sentences) <= window_size:
        if not sentences:
            return []
        return [" ".join(sentences)]
        
    i = 0
    step = max(1, window_size - overlap)
    while i < len(sentences):
        chunk_sentences = sentences[i:i+window_size]
        chunks.append(" ".join(chunk_sentences))
        i += step
    return chunks


def chunk_recursive(text, max_length=150):
    # A simplified recursive chunker
    chunks = []
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    for p in paragraphs:
        if len(p) <= max_length:
            chunks.append(p)
        else:
            sentences = [s + '.' for s in p.split('. ') if s.strip()]
            current_chunk = ""
            for s in sentences:
                if len(current_chunk) + len(s) <= max_length:
                    current_chunk += s + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = s + " "
            if current_chunk:
                chunks.append(current_chunk.strip())
    return chunks

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def setup_databases(_model):
    db = lancedb.connect("memory://")
    
    data_dir = "data"
    all_chunks_fixed = []
    all_chunks_window = []
    all_chunks_recursive = []
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # generate chunks for this file
                f_chunks_fixed = chunk_fixed_size(content, 250)
                f_chunks_window = chunk_sliding_window(content, window_size=3, overlap=1)
                f_chunks_recursive = chunk_recursive(content, 250)
                
                # add filename info to display
                all_chunks_fixed.extend([{"text": c, "filename": filename} for c in f_chunks_fixed])
                all_chunks_window.extend([{"text": c, "filename": filename} for c in f_chunks_window])
                all_chunks_recursive.extend([{"text": c, "filename": filename} for c in f_chunks_recursive])
                
    if not all_chunks_fixed:
        st.error("No text data found in the `data/` directory.")
        return None, None, None
        
    data_fixed = [{"text": c["text"], "filename": c["filename"], "vector": _model.encode(c["text"]).tolist()} for c in all_chunks_fixed]
    data_window = [{"text": c["text"], "filename": c["filename"], "vector": _model.encode(c["text"]).tolist()} for c in all_chunks_window]
    data_recursive = [{"text": c["text"], "filename": c["filename"], "vector": _model.encode(c["text"]).tolist()} for c in all_chunks_recursive]
    
    tbl_fixed = db.create_table("fixed_chunks", data=pd.DataFrame(data_fixed))
    tbl_window = db.create_table("window_chunks", data=pd.DataFrame(data_window))
    tbl_recursive = db.create_table("recursive_chunks", data=pd.DataFrame(data_recursive))
    
    return tbl_fixed, tbl_window, tbl_recursive

with st.spinner("Initializing models and chunking files from `data/`..."):
    model = load_model()
    tbl_fixed, tbl_window, tbl_recursive = setup_databases(model)

st.divider()

if tbl_fixed is None:
    st.stop()

# Search UI
query = st.text_input("Enter a search query:", placeholder="e.g., 'What is machine learning?'", help="Type a question or term.")

if query:
    query_vec = model.encode(query).tolist()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔲 Fixed Size Results")
        res_fixed = tbl_fixed.search(query_vec).limit(3).to_pandas()
        for idx, row in res_fixed.iterrows():
            with st.expander(f"Result {idx+1} | {row.get('filename', 'Unknown')} (Dist: {row['_distance']:.4f})", expanded=True):
                st.write(f"**Chunk Text:**")
                st.code(row['text'], language="text")
                
    with col2:
        st.subheader("🌳 Recursive Results")
        res_recur = tbl_recursive.search(query_vec).limit(3).to_pandas()
        for idx, row in res_recur.iterrows():
            with st.expander(f"Result {idx+1} | {row.get('filename', 'Unknown')} (Dist: {row['_distance']:.4f})", expanded=True):
                st.write(f"**Chunk Text:**")
                st.info(row['text'])
                
    with col3:
        st.subheader("🪟 Sliding Window Results")
        res_window = tbl_window.search(query_vec).limit(3).to_pandas()
        for idx, row in res_window.iterrows():
            with st.expander(f"Result {idx+1} | {row.get('filename', 'Unknown')} (Dist: {row['_distance']:.4f})", expanded=True):
                st.write(f"**Chunk Text:**")
                st.success(row['text'])
