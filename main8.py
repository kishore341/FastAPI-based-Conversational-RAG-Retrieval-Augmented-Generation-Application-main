from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import re
import time
import asyncio
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# LangChain imports
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document

import llm_client
from utils import extract_text_async, store_file_in_db, fetch_files_from_db
from llm_client import llm, embeddings

load_dotenv()

# Global variables
vectorstore = None
chat_store = {}

# Lifespan event to initialize FAISS vectorstore
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore
    try:
        faiss_index_path = r"C:\Users\zyxan\PycharmProjects\Reading_documents_from_Mongo_DB_Atlas_with_FastAPI\faiss_index"
        index_file = os.path.join(faiss_index_path, "index.faiss")

        if os.path.exists(index_file):
            vectorstore = FAISS.load_local(
                folder_path=faiss_index_path,
                embeddings=llm_client.embeddings,
                allow_dangerous_deserialization=True
            )
            print("[LIFESPAN] FAISS vectorstore loaded successfully.")
        else:
            print(f"[LIFESPAN WARNING] No FAISS index found at {index_file}. Creating a new one.")

            from langchain_core.documents import Document
            dummy_doc = Document(page_content="init")
            vectorstore = FAISS.from_documents([dummy_doc], embedding=llm_client.embeddings)

            os.makedirs(faiss_index_path, exist_ok=True)
            vectorstore.save_local(faiss_index_path)
            print("[LIFESPAN] New FAISS vectorstore created and saved.")

    except Exception as e:
        print(f"[LIFESPAN ERROR] {e}")
        vectorstore = None

    yield

app = FastAPI(lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background file processing
async def process_files_in_background(session_id, file_data_list):
    global vectorstore  # Ensure we modify the global FAISS instance
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for file_name, file_bytes in file_data_list:
        ext = os.path.splitext(file_name)[1].lower()
        raw_text = await extract_text_async(ext, file_bytes)

        # Store in MongoDB
        store_file_in_db(file_name, raw_text, session_id, ext, embeddings_model=embeddings)

        # ➕ Convert to LangChain documents
        docs = text_splitter.create_documents([raw_text])
        for doc in docs:
            doc.metadata["source"] = file_name

        # ➕ Add to FAISS vectorstore
        if vectorstore:
            vectorstore.add_documents(docs)
            # Optionally save updated vectorstore
            vectorstore.save_local(r"C:\Users\zyxan\PycharmProjects\Reading_documents_from_Mongo_DB_Atlas_with_FastAPI\faiss_index")
@app.post("/upload")
async def upload_files(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    start_time = time.time()
    try:
        file_data_list = []
        for file in files:
            content = await file.read()
            file_data_list.append((file.filename, content))

        asyncio.create_task(process_files_in_background(session_id, file_data_list))
        print(f"[UPLOAD] Files are being processed for session: {session_id}")
        return {"status": "success", "message": "Files are uploaded and processing."}

    except Exception as e:
        print(f"[UPLOAD ERROR] {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        print(f"[UPLOAD TIME] {time.time() - start_time:.2f} seconds")

# Utility to clean markdown
def clean_markdown(text: str) -> str:
    return re.sub(r'[*_`]', '', text)

@app.post("/ask")
async def ask_question(
    session_id: str = Form(...),
    question: str = Form(...)
):
    try:
        db_files = fetch_files_from_db(session_id)
        if not db_files:
            return JSONResponse(status_code=404, content={"message": f"No files found for session {session_id}"})

        start_time = time.time()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = []
        for file_name, content in db_files:
            docs = text_splitter.create_documents([content])
            for doc in docs:
                doc.metadata["source"] = file_name
                splits.append(doc)

        if not splits:
            raise HTTPException(status_code=400, detail="No content available for the query.")

        if vectorstore is None:
            raise HTTPException(status_code=500, detail="Vectorstore not initialized.")

        retriever = vectorstore.as_retriever()

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, formulate a standalone version of the question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant. Use ONLY the provided context to answer the question: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        session_history = chat_store.setdefault(session_id, ChatMessageHistory())

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda _: session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        response = conversational_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )

        if not isinstance(response, dict) or not response.get("answer"):
            raise HTTPException(status_code=500, detail="No answer found.")

        raw_answer = re.sub(r"<think>.*?</think>", "", response['answer'], flags=re.DOTALL).strip()
        answer = clean_markdown(raw_answer)

        print(f"[ASK RESPONSE] Session: {session_id}\nQuestion: {question}\nAnswer: {answer}\n")

        return {"answer": answer}

    except Exception as e:
        print(f"[ASK ERROR] {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        print(f"[ASK TIME] {time.time() - start_time:.2f} seconds")
