# rag.py
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.schema import Document
from PyPDF2 import PdfReader
import streamlit as st

PERSIST_DIR = "./rag_db"

def load_documents(files):
    """
    Accepts list of files and returns list of Document objects.
    Supports PDF and text files.
    """
    documents = []
    for file in files:
        try:
            if file.type == "application/pdf":
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        documents.append(Document(page_content=text.strip()))
            else:
                content = file.getvalue().decode("utf-8")
                if content.strip():
                    documents.append(Document(page_content=content.strip()))
        except Exception as e:
            st.warning(f"Failed to read {file.name}: {e}")
    return documents


def build_rag_chain(documents):
    """
    Takes list of Document, builds embeddings, vectorstore, and RAG chain.
    Returns the rag_chain and retriever.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    os.makedirs(PERSIST_DIR, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    vectorstore.persist()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = """
    Using the following pieces of retrieved context, answer the question comprehensively and concisely.
    Ensure your response fully addresses the question based on the given context.

    **IMPORTANT:**
    Just provide the answer and never mention or refer to having access to the external context or information in your answer.
    If you are unable to determine the answer from the provided context, state 'I don't know.'

    Question: {question}
    Context: {context}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOllama(model="llama3")  # Replace "llama3" with your actual model

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

