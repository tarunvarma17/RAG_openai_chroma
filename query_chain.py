from chunk_text import *
from embedding_indexing import *
from langchain_community.llms import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

llm = OpenAI(model_name = 'gpt-3.5-turbo')

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Every answer should be in the form of a 4 line poem with AABB rhyme scheme.
{context}
Question: {input}""")


def get_answer_chain(index, query):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = index.as_retriever(search_kwargs={"k": 1})
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({"input":query})
    return response["answer"]