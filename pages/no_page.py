import faiss
from langchain.vectorstores import FAISS
import streamlit as st
import pickle
from langchain.embeddings import HuggingFaceInstructEmbeddings
from streamlit_lottie import st_lottie
import json
import os

def load_lottiefile(filepath: str):
    """Gets the file path and returns the file in dict format"""
    with open(filepath, "r", encoding="utf8") as f:
        return json.load(f)

def UI_communication(value):
      """Gets the String and displays it with the animation"""
      st.markdown(f"<h4 style='text-align: center;'>{value}</h4>", unsafe_allow_html=True)
      lottie_streamlit = load_lottiefile("/content/Animation - 1711522865345.json")
      st.lottie(lottie_streamlit, speed=1.0, reverse = False, height=200)

col1, col2 = st.columns([0.85,0.15])
with col1:
  # st.title(":rainbow[Resume Semantic Searcher :mag_right:]")
  st.markdown('## :rainbow[Resume Semantic Searcher :mag_right:]', unsafe_allow_html=True)
with col2: 
  if st.button("Home"):
    # streamlit_js_eval(js_expressions="parent.window.location.reload()")
    st.switch_page("app.py")

placeholder = st.empty()
with placeholder.container():
    UI_communication("Loading...")

# Load the index from the file
index = faiss.read_index("/content/faiss_index.bin")

# Load metadata
with open("/content/docstore.pkl", "rb") as f:
    docstore = pickle.load(f)

with open("/content/index_to_docstore_id.pkl", "rb") as f:
    index_to_docstore_id = pickle.load(f)

# Initialize the embeddings
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"})

# Create a FAISS vector store from the loaded index
db_instructEmbedd = FAISS(
    embedding_function=instructor_embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": 10})
placeholder.empty()
#Get input
query = st.text_input("Enter your query:")

if query:
    with placeholder.container():
      UI_communication("Retrieving documents...")
    #document retrieval
    docs = retriever.get_relevant_documents(query)
    placeholder.empty()
    for idx, doc in enumerate(docs):
        pdf_path = doc.metadata["source"]
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        st.download_button(
            label=f"Download {os.path.basename(pdf_path)}",
            data=pdf_data,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf",
            key=f"download_button_{idx}"
        )
