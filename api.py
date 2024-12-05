from dotenv import load_dotenv
import os
import re
import redis
import pickle

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.chat_models import ChatOpenAI

# Load api keys
def load_api_keys():
    load_dotenv()
    global pinecone_api_key
    global openai_api_key
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

# Load documents
def load_docs():
    pdf_link = "https://arxiv.org/pdf/2402.16893"
    loader = PyPDFLoader(file_path=pdf_link,
                        extract_images=False)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 100)
    pages = loader.load_and_split(text_splitter)
    return pages

# Load model
def load_model(api_key):
    model = ChatOpenAI(openai_api_key=api_key,
                       model="gpt-3,5-turbo",
                       temperature=0.2)
    return model

# Load embeddings
def load_embeddings(api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return embeddings

def initialize_api():
    global model, embeddings, docs
    print("load everything")
    load_api_keys()
    model = load_model(api_key=openai_api_key)
    embeddings = load_embeddings(api_key=openai_api_key)
    docs = load_docs()

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_api()
    try:
        yield
    finally:
        print("shutting down api...")


app = FastAPI(
    title = "API",
    description = "RAG api",
    lifespan = lifespan
)



# Homepage
@app.get("/")
async def root():
    return {"message":"API is running."}

# Read doc
@app.get("/page/{index}")
async def get_page(index:int):
    if docs is None:
        raise(HTTPException(status_code=500, detail="Data not loaded, please try again later."))

    if index<0 or index>=len(docs):
        raise(HTTPException(status_code=404, detail="Index out of range."))
    return {"index": index, "content": re.sub("\s\s+", "", docs[index].page_content.replace('\n','').strip())}
