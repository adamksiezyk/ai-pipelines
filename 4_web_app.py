import itertools
import os
import streamlit as st
from glob import glob
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def format_docs(docs):
    docs_format = []
    for i, doc in enumerate(docs, 1):
        header = f"[{i}]"
        if "title" in doc.metadata and doc.metadata["title"]:
            header += f" Title: {doc.metadata['title'].strip()}"
        if "author" in doc.metadata and doc.metadata["author"]:
            header += f" Author: {doc.metadata['author'].strip()}"
        if "source" in doc.metadata and doc.metadata["source"]:
            header += f" File: {os.path.split(doc.metadata['source'])[-1].strip()}"
        if "page" in doc.metadata and doc.metadata["page"]:
            header += f" Page: {doc.metadata['page']}"
        doc_formatted = header + "\n\n" + doc.page_content.strip()
        docs_format.append(doc_formatted)
    return "\n\n".join(docs_format)


if "retrieval_chain" not in st.session_state:
    # Laod documents
    docs = [PyMuPDFLoader(fp).load() for fp in glob("./documents/*.pdf")]
    print(f"Found {len(docs)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", ".\n", ".", " ", "\n"]
    )
    chunks = text_splitter.split_documents(itertools.chain.from_iterable(docs))
    
    # Create Vector Store
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    
    # Retrieval chain
    retriever = vectorstore.as_retriever()
    prompt = PromptTemplate.from_template("[CONTEXT]\n{context}\n[/CONTEXT]")
    st.session_state["retrieval_chain"] = retriever | format_docs | prompt

# Chat
chat_llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dev",
    model="mistral",
    temperature=0,
    max_tokens=256,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": """You are an AI assistant, that helps people find information in documents.
Use the following pieces of context enclosed between [CONTEXT] and [/CONTEXT] tags to answer the question.
Use three sentences maximum and keep the answer concise. If you don't know the answer, just say that you don't know.
""",
        }
    ]

st.title("🚀 AI pipelines Web App")
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Pass your input here")
if prompt:
    st.chat_message("user").markdown(prompt)
    if "context" not in st.session_state:
        context = st.session_state["retrieval_chain"].invoke(prompt).text
        st.session_state["context"] = context
        st.chat_message("user").markdown(context)
        st.session_state.messages.append({"role": "user", "content": context})
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chat_llm.stream(st.session_state.messages)

    complete_response = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        for chunk in response:
            if chunk.content is not None:
                complete_response += chunk.content
                message_placeholder.markdown(complete_response + "▌")
                message_placeholder.markdown(complete_response)
    st.session_state.messages.append(
        {"role": "assistant", "content": complete_response}
    )