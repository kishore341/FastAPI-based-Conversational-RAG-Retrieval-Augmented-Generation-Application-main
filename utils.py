import os
import pdfplumber
import docx
import pandas as pd
from PIL import Image
import pytesseract
import io
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import List

# Load environment variables
load_dotenv()

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]
collection = db["reading_files"]

# Tesseract path (adjust as needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Chunking
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# ✅ Store chunks with vector embeddings
def store_file_in_db(file_name, file_content, session_id, file_type, embeddings_model):
    chunks = split_text_into_chunks(file_content)
    for idx, chunk in enumerate(chunks):
        try:
            vector = embeddings_model.embed_query(chunk)
            document = {
                "file_name": file_name,
                "chunk_id": idx,
                "chunk_content": chunk,
                "chunk_content_vector": vector,  # ✅ Vector stored here
                "session_id": session_id,
                "file_type": file_type,
                "upload_date": datetime.utcnow()
            }
            result = collection.insert_one(document)
            print(f"[DB INSERTED] ID: {result.inserted_id}")
        except Exception as e:
            print(f"[DB ERROR] {e}")

# Extract text
async def extract_text_async(ext, content):
    try:
        if ext == ".pdf":
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "".join(page.extract_text() or "" for page in pdf.pages)
        elif ext == ".docx":
            doc = docx.Document(io.BytesIO(content))
            return "\n".join(para.text for para in doc.paragraphs)
        elif ext in [".png", ".jpg", ".jpeg"]:
            image = Image.open(io.BytesIO(content))
            return pytesseract.image_to_string(image)
        elif ext == ".csv":
            df = pd.read_csv(io.BytesIO(content))
            return df.to_string(index=False)
        elif ext == ".txt":
            return content.decode("utf-8")
        else:
            raise ValueError("Unsupported file type.")
    except Exception as e:
        print(f"[EXTRACT ERROR] {e}")
        raise

# Fetch files
def fetch_files_from_db(session_id: str):
    try:
        files = collection.find({"session_id": session_id}).sort("upload_date", 1)
        return [(file["file_name"], file["chunk_content"]) for file in files]
    except Exception as e:
        print(f"[FETCH ERROR] {e}")
        return []
