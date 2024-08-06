import pickle
from chunk_text import *
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()
EMBEDDINGS_FOLDER_PATH = '/home/tarunvarma17/RAG_chroma/Embeddings'

def indexing(filepath):
    filename = filepath.split("/")[-1][:-4]
    docs = prepare(filepath)

    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')
    
    if os.path.exists(f'{EMBEDDINGS_FOLDER_PATH}/{filename}'):
        vectorstore = Chroma(persist_directory=f'{EMBEDDINGS_FOLDER_PATH}/{filename}', embedding_function=embeddings)
        print("Embeddings stored from disk")
    else:
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=f'{EMBEDDINGS_FOLDER_PATH}/{filename}')
        print("Embeddings computed")
    return vectorstore


# filepath = '/home/tarunvarma17/RAG_chroma/PDF_Files/rag_steps.pdf'
# indexing(filepath)