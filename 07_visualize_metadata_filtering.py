import os
import re
import streamlit as st
import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer

# Page Config
st.set_page_config(page_title="Metadata Filtering Demo", page_icon="🏷️", layout="wide")
st.title("Metadata Filtering Demo")

st.markdown("""
This application demonstrates **Semantic Search with Metadata Filtering**. 
We process the text documents and automatically extract a metadata attribute (`exam = True/False`) if the document explicitly requires an exam to pass ("Forma zaliczenia Egzamin").
""")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def setup_databases(_model):
    db = lancedb.connect("memory://")
    
    data_dir = "data"
    all_chunks = []
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract metadata feature
                is_mandatory = "Course status\nMandatory" in content or "Course status Mandatory" in content
                has_exam = "Assessment form\nExam" in content or "Assessment form Exam" in content

                term_match = re.search(r'(?i)semester\s*(\d+)', content)
                term = int(term_match.group(1)) if term_match else 0

                # add filename, text content, and metadata
                all_chunks.append({
                    "text": content, 
                    "filename": filename,
                    "exam": has_exam,
                    "term": term,
                    "mandatory": is_mandatory
                })
                
    if not all_chunks:
        st.error("No text data found in the `data/` directory.")
        return None
        
    data_with_embeddings = [
        {
            "text": c["text"], 
            "filename": c["filename"], 
            "exam": c["exam"],
            "term": c["term"],
            "mandatory": c["mandatory"],
            "vector": _model.encode(c["text"]).tolist()
        } 
        for c in all_chunks
    ]
    
    tbl = db.create_table("documents", data=pd.DataFrame(data_with_embeddings))
    return tbl

with st.spinner("Initializing model and extracting metadata..."):
    model = load_model()
    tbl = setup_databases(model)

st.divider()

if tbl is None:
    st.stop()

# Sidebar Filters
st.sidebar.header("Filter Options")
df_all = tbl.to_pandas()
available_terms = sorted(df_all['term'].unique().tolist())

selected_terms = st.sidebar.multiselect(
    "Select Term(s)", 
    options=available_terms, 
    default=available_terms
)

mandatory_filter = st.sidebar.radio(
    "Course Type", 
    ["All", "Mandatory Only", "Optional Only"]
)

# Search UI
query = st.text_input("Enter a search query:", placeholder="e.g., 'Zasady zaliczania przedmiotu'", help="Type a question or term.")

if query:
    query_vec = model.encode(query).tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Standard Semantic Search")
        st.caption("Searching across **all** documents regardless of metadata.")
        
        res_all = tbl.search(query_vec).limit(5).to_pandas()
        
        if res_all.empty:
            st.info("No results found.")
        else:
            for idx, row in res_all.iterrows():
                exam_label = "Exam: Yes" if row['exam'] else "Exam: No"
                term_label = f"Term: {row['term']}"
                mand_label = "Mandatory" if row['mandatory'] else "Optional"
                
                with st.expander(f"📄 {row['filename']} (Dist: {row['_distance']:.4f}) | {exam_label} | {term_label} | {mand_label}", expanded=True):
                    st.write(f"**Document Text:**")
                    st.info(row['text'])
                
    with col2:
        st.subheader("🏷️ Filtered Search")
        
        filters = []
        if selected_terms:
            terms_str = ", ".join(str(t) for t in selected_terms)
            filters.append(f"term IN ({terms_str})")
        else:
            filters.append("term IN (-1)") # force no match
            
        if mandatory_filter == "Mandatory Only":
            filters.append("mandatory = true")
        elif mandatory_filter == "Optional Only":
            filters.append("mandatory = false")
            
        filter_str = " AND ".join(filters)
        
        st.caption(f"Applying LanceDB pre-filter: `{filter_str}`")
        
        if filter_str:
            res_filtered = tbl.search(query_vec).where(filter_str).limit(5).to_pandas()
        else:
            res_filtered = tbl.search(query_vec).limit(5).to_pandas()
        
        if res_filtered.empty:
            st.warning("No results found that matched the filters.")
        else:
            for idx, row in res_filtered.iterrows():
                exam_label = "Exam: Yes" if row['exam'] else "Exam: No"
                term_label = f"Term: {row['term']}"
                mand_label = "Mandatory" if row['mandatory'] else "Optional"
                
                with st.expander(f"📄 {row['filename']} (Dist: {row['_distance']:.4f}) | {exam_label} | {term_label} | {mand_label}", expanded=True):
                    st.write(f"**Document Text:**")
                    st.success(row['text'])
