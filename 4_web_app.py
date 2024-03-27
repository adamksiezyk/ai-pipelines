import itertools
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
    return "\n\n".join(doc.page_content for doc in docs)


# Laod documents
docs = [PyMuPDFLoader(fp).load() for fp in glob("./documents/*")]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", ".\n", ".", " ", "\n"]
)
chunks = text_splitter.split_documents(itertools.chain.from_iterable(docs))

# Create Vector Store
embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

# RAG chain
retriever = vectorstore.as_retriever()
prompt = PromptTemplate.from_template("""Question: {question} 

Context: {context}""")
retrieval_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
)

# Chat
chat_llm = ChatOpenAI(
    base_url="http://localhost:9999/v1",
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
Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
""",
        }
    ]

st.title("🚀 AI pipelines Web App")
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Pass your input here")
if prompt:
    st.chat_message("user").markdown(prompt)
    prompt_with_context = retrieval_chain.invoke(prompt).text
    st.session_state.messages.append({"role": "user", "content": prompt_with_context})

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