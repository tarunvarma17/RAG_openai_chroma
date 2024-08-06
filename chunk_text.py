import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text(path):
    loader = PyPDFLoader(path)
    raw_text = loader.load()
    return raw_text

def chunk_text_langchain(docs, chunk_size = 1000, chunk_overlap = 200):
    splitter =  RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = splitter.split_documents(docs)
    return chunks

def prepare(path, method = "langchain"):
    raw_text = extract_text(path)
    docs = chunk_text_langchain(raw_text)
    return docs

# filepath = '/home/tarunvarma17/RAG_chroma/PDF_Files/rag_steps.pdf'

# docs = prepare(filepath)
# for c in docs:
#     print(c)
#     print("-----------------------------------------------------")