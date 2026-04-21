# llm_client.py
import os
from dotenv import load_dotenv

#from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ gor GROQ
from langchain_huggingface import HuggingFaceEmbeddings  # for sambanova

from langchain_openai import ChatOpenAI


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
#sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")




#llm = ChatOpenAI(
#   base_url="https://api.sambanova.ai/v1",  # 🔁 Replace with SambaNova's actual base URL
#    api_key=sambanova_api_key,
#    model="DeepSeek-R1",  # 🔁 Replace with the correct model name
#    temperature=0,
#)

# ✅ Define the LLM for use
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key,
    model="llama3-70b-8192",
   temperature=0,
)

# ✅ Define the embeddings object
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
