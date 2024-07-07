from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
import pickle
import faiss
from langchain.vectorstores import FAISS
import streamlit as st
from langchain.embeddings import HuggingFaceInstructEmbeddings
import os
from streamlit_lottie import st_lottie
import json

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

class CustomPyPDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def lazy_load(self):
        try:
            # Load the PDF file using PyPDFLoader
            documents = PyPDFLoader(self.file_path).load()
            return documents
        except Exception as e:
            print(f"Error loading {self.file_path}: {str(e)}")
            return []

#Get directory
user_dir = st.text_input("Enter the folder path")

if user_dir:
    placeholder = st.empty()
    with placeholder.container():
      UI_communication("Loading Data...")

    # Use DirectoryLoader with the custom loader
    loader = DirectoryLoader(user_dir, glob="./*.pdf", loader_cls=CustomPyPDFLoader)

    # Load the documents
    documents = list(loader.load())

    placeholder.empty()
    with placeholder.container():
      UI_communication("Splitting texts...")

    #splitting the text into
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    #Initialize embeddings 
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                          model_kwargs={"device": "cuda"})

    placeholder.empty()
    with placeholder.container():
      UI_communication("Embedding to vectordb...")
    #Store the embedded documents to FAISS vectordb
    db_instructEmbedd = FAISS.from_documents(texts, instructor_embeddings)
    retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": 10})

    placeholder.empty()
    #Get Input
    query = st.text_input("Enter your query:")

    if query:
        # Retrieve documents
        with placeholder.container():
            UI_communication("Retrieving documents...")
        docs = retriever.get_relevant_documents(query)
        placeholder.empty()
        for idx, doc in enumerate(docs):
            pdf_path = doc.metadata["source"]
            # st.write(pdf_path)
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()
            st.download_button(
                label=f"Download {os.path.basename(pdf_path)}",
                data=pdf_data,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
                key=f"download_button_{idx}"
            )


