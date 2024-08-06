import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader
from query_chain import *
from query import *
from embedding_indexing import *
from chunk_text import *

UPLOAD_FOLDER = '/home/tarunvarma17/RAG_chroma/PDF_Files'

def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None


def main():
    
    st.header("PDF RAG")
    pdf = st.file_uploader("Upload pdf", type = 'pdf')
    if pdf is not None:
        filepath = save_uploaded_file(pdf)
        
        index = indexing(filepath)
        
        query = st.text_input("Ask question here:")
        if query:
            
            # context = get_top_results(query, filepath, 2)

            # result = get_answer(context, query)
            
            result = get_answer_chain(index, query)
            st.write(result)
            

if __name__ == '__main__':
    main()