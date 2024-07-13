import langchain.schema
import langchain.schema.document
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
import os
import re
import getpass
from dotenv import load_dotenv


def get_pdf(pdf_docs):
    documents = []
    for pdf in pdf_docs:
        doc = PdfReader(pdf)
        name = pdf.name.split(".")[0]
        page_count = 1
        for page in doc.pages:
            documents.append(
                Document(
                    page_content=(name + ": " + page.extract_text()),
                    metadata={"source": name, "page": page_count},
                )
            )
            page_count += 1
    return documents


def get_chunks(raw_text):
    split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = split.split_documents(raw_text)
    for text in text_chunks:
        text.page_content = text.metadata["source"] + ": " + text.page_content
    return text_chunks


def conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", convert_system_message_to_human=True
    )
    prompt_search_query = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "Given the above chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.",
            ),
            ("user", "{input}"),
        ]
    )
    prompt_get_answer = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Answer briefly and keep the answer concise.  \n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    retriever = vectorstore.as_retriever()
    retriever_chain = create_history_aware_retriever(
        llm, retriever, prompt_search_query
    )
    document_chain = create_stuff_documents_chain(llm, prompt_get_answer)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    return retrieval_chain


def process(pdf_docs):
    raw_text = get_pdf(pdf_docs)
    text_chunks = get_chunks(
        raw_text
    )  # Intel/neural-embedding-v1, maidalun1020/bce-embedding-base_v1, Alibaba-NLP/gte-Qwen1.5-7B-instruct,Linq-AI-Research/Linq-Embed-Mistral, mixedbread-ai/mxbai-embed-large-v1

    embedding = HuggingFaceInferenceAPIEmbeddings(
        api_key="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",
        model_name="maidalun1020/bce-embedding-base_v1",
    )
    vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embedding)
    return vectorstore


def get_response(input):
    response = st.session_state.conversation.invoke(
        {
            "chat_history": st.session_state.history,
            "input": input,
        }
    )
    return response


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "message" not in st.session_state:
        st.session_state.message = []
    if "history" not in st.session_state:
        st.session_state.history = []
    st.header("Chat with multiple PDFs:")
    with st.sidebar:
        st.subheader("Your Document")
        pdf_docs = st.file_uploader("Upload your pdfs here", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                vectorstore = process(pdf_docs)
                st.session_state.conversation = conversation_chain(vectorstore)

    input = st.chat_input("Write Something")
    response = None
    if input:
        st.session_state.message.append({"role": "user", "content": input})
        response = get_response(input)
        st.session_state.history.append({"role": "user", "content": input})
    if response:
        st.session_state.history.append({"role": "ai", "content": response["answer"]})
        response["answer"] += "  \n  \nSource:  \n"
        for source in response["context"]:
            # print(source.metadata['source'])
            response["answer"] += (
                source.metadata["source"]
                + " ("
                + "page: "
                + str(source.metadata["page"])
                + ")"
            )
            response["answer"] += "  \n"
        st.session_state.message.append({"role": "ai", "content": response["answer"]})

    for message in st.session_state.message:
        with st.chat_message(message["role"]):
            st.write(message["content"])


if __name__ == "__main__":
    main()
