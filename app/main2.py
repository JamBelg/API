from dotenv import load_dotenv
import os
import re
import time
# cache memory
from cachetools import cached, TTLCache
from functools import lru_cache

from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeVectorStore

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from pydantic import BaseModel

# langchain (openai, prompts, vectorestore)
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap

# Set up TTL cache (Time-to-Live Cache)
cache_docs = TTLCache(maxsize=10, ttl=3600)  # Cache for 1 hour

pdf_list = ["https://arxiv.org/pdf/2402.16893",
                "https://arxiv.org/pdf/2312.10997",
                "https://arxiv.org/pdf/2408.10343",
                "https://arxiv.org/pdf/2409.14924",
                "https://arxiv.org/pdf/2406.07348"]

# Load api keys
@lru_cache(maxsize=1)
def load_api_keys():
    load_dotenv()
    global pinecone_api_key
    global openai_api_key
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

# Load documents
@cached(cache_docs)
def load_docs():
    pdf_list = ["https://arxiv.org/pdf/2402.16893",
                "https://arxiv.org/pdf/2312.10997",
                "https://arxiv.org/pdf/2408.10343",
                "https://arxiv.org/pdf/2409.14924",
                "https://arxiv.org/pdf/2406.07348"]
    docs = []
    for pdf_link in pdf_list:
        loader = PyPDFLoader(file_path=pdf_link,
                            extract_images=False)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 100)
        pages = loader.load_and_split(text_splitter)
        docs.extend(pages)
    return pdf_list, docs

# Load model
@lru_cache(maxsize=1)
def load_model(api_key):
    model = ChatOpenAI(openai_api_key=api_key,
                       model="gpt-4",
                       temperature=0.2)
    return model

# Load embeddings
@lru_cache(maxsize=1)
def load_embeddings(api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return embeddings

# Prompt
@lru_cache(maxsize=1)
def define_prompt():
    # Prompt template (context + question)
    template = """
    Answer the question based on the content below. If you can't answer the question, reply 'I don't know'.
    Context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

# Load retriever
# Load retriever
def load_retriever(api_key, embeddings):
    index_name = "pdf-files"

    # Initialize Pinecone client
    pinecone_client = Pinecone(
        api_key=api_key
    )

    # Check if index exists
    if index_name not in pinecone_client.list_indexes().names():
        raise ValueError(f"Index '{index_name}' does not exist in Pinecone. Ensure it is created and populated.")

    # Connect to existing Pinecone index
    vectorstore = PineconeVectorStore(
        pinecone_api_key=api_key,
        embedding=embeddings,
        index_name=index_name
    )

    # Set up retriever to fetch top 3 documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

# Load parser
@lru_cache(maxsize=1)
def load_parser():
    parser = StrOutputParser()
    return parser

def initialize_api():
    global model, embeddings, pdf_list, docs, retriever, prompt, parser
    load_api_keys()
    model = load_model(api_key=openai_api_key)
    embeddings = load_embeddings(api_key=openai_api_key)
    pdf_list = pdf_list
    #pdf_list, docs = load_docs()
    prompt = define_prompt()
    retriever = load_retriever(api_key=pinecone_api_key,
                               embeddings=embeddings)
    parser = load_parser()
    print("load everything")


@asynccontextmanager
async def lifespan(app: FastAPI):
    time1 = time.time()
    initialize_api()
    print("Time elapsed loading api: %.2f seconds" % (time.time()-time1))
    try:
        yield
    finally:
        print("shutting down api...")


app = FastAPI(
    title = "API",
    description = "RAG api",
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to restrict specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Homepage
@app.get("/")
async def root():
    return {"message":"API is running."}

# API info
@app.get("/info")
async def get_info():
    return {"Info":"API for chatbot using RAG about PDF documents (also about RAG)",
            "Python":"3.1",
            "Workflow":"FastAPI",
            "Context docs": pdf_list,
            "Docs split": "chunks of size 1000 with averlaps of 100",
            "Chunks count": len(docs),
            "LLM":"GPT-4",
            "Retriever":"Pinecone",
            "Retriever metric":"Cosine distance",
            "Retriever relevant docs":"3"}

# Read doc
@app.get("/page/{index}")
async def get_page(index:int):
    if docs is None:
        raise(HTTPException(status_code=500, detail="Data not loaded, please try again later."))

    if index<0 or index>=len(docs):
        raise(HTTPException(status_code=404, detail="Index out of range."))
    return {"index": index, "content": re.sub("\s\s+", "", docs[index].page_content.replace('\n','').strip())}

class QuestionRequest(BaseModel):
    question: str

class RAGresponse(BaseModel):
    question: str
    response: str
    time: str

@app.post("/ask", response_model=RAGresponse)
async def ask_question(request:QuestionRequest):
    try:
        start_time = time.time()
        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(request.question)

        # Combine the text of the documents into a single string
        context = " ".join([doc.page_content for doc in relevant_docs])

        # Create the chain with Runnable components
        chain = (
            RunnableMap({
                "context" : RunnablePassthrough(),
                "question": RunnablePassthrough()
            })
            | prompt
            | model
            | parser
        )

        result = chain.invoke({"question": request.question, "context": context})
        exec_time = "%.2f seconds" % (time.time()-start_time)

        return {
            "question": request.question,
            "response": result,
            "time": exec_time,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the question: {str(e)}")