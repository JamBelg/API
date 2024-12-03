import os
from typing import List, Dict, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, Path as PathParam, Query, File, UploadFile
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from contextlib import asynccontextmanager

# Define the directory containing PDF files
PDF_DIRECTORY = "C:/Users/ch1jbelgac1/Work Folders/MyDocuments/localRAG/pdf_files"

# Load and process the PDF files during startup
processed_pages: Optional[List[Dict]] = None

OPENAI_API_KEY = "api_key"
PINECONE_API_KEY = "pinecone_api_key"

# Function to read and split PDFs into chunks
def read_split_pdf(pdf_directory: str) -> List[Dict]:
    # Define the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Initialize a list to hold all the processed pages
    doc_splits = []

    # Process each PDF file in the directory
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):  # Ensure only PDFs are processed
            file_path = os.path.join(pdf_directory, filename)
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split(text_splitter)  # Split the PDF into chunks
            doc_splits.extend(pages)  # Add the chunks to the combined list
    return doc_splits


def embeddings():
    from langchain_openai.embeddings import OpenAIEmbeddings
    embeddings =OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return embeddings

def vectorestores(pages, embeddings):
    # Pinecone for cosine distance between question and different documents
    index_name = "pdf-files"
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

    pinecone = PineconeVectorStore.from_documents(
        pages, embedding=embeddings, index_name=index_name
    )
    return pinecone

def define_prompt():
    # Prompt template (context + question)
    from langchain.prompts import ChatPromptTemplate
    template = """
    Answer the question based on the content below. If you can't answer the question, reply 'I don't know'.
    Context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def define_parser():
    # Parser
    from langchain_core.output_parsers import StrOutputParser
    parser = StrOutputParser()
    return parser

def initialize_model():
    from langchain_openai.chat_models import ChatOpenAI
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                    model="gpt-3.5-turbo",
                    temperature=0.2)
    return model
    

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processed_pages
    global prompt
    global parser
    global model
    processed_pages = read_split_pdf(PDF_DIRECTORY)
    prompt = define_prompt()
    parser = define_parser()
    model = initialize_model()
    yield
    # print(f"Loaded {len(processed_pages)} chunks from PDFs in '{PDF_DIRECTORY}'.")

# Initialize the app
app = FastAPI(
    title="RAG_APP",
    description="Retrival Augmented Generation APP which let's user upload a file and get the answer for the question using LLMs",
    lifespan=lifespan
)




@app.get("/")
async def root():
    """
    Root endpoint to verify that the API is running.
    """
    return {"message": "API is running. \nUse /page/{index} to fetch a page.\nUse /predict?query={question} to get a response."}

@app.get("/page/{index}")
async def get_page(index: int):
    """
    Endpoint to fetch the page content at the given index.
    """
    global processed_pages
    if processed_pages is None:
        raise HTTPException(status_code=500, detail="Data not loaded. Please try again later.")

    # Ensure the index is within range
    if index < 0 or index >= len(processed_pages):
        raise HTTPException(status_code=404, detail="Index out of range.")

    # Return the content of the requested page
    return {"index": index, "content": processed_pages[index]}

# Define the request and response schema
class QuestionRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    question: str
    answer: str
    context: List[str]

@app.post("/ask", response_model=RAGResponse)
async def ask_question(pinecone, prompt, model, parser, request: QuestionRequest):
    """
    Endpoint to accept a question and return a RAG-based answer.
    """
    try:
        chain = (
            {"context":pinecone.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | model
            | parser
        )
        result = chain.invoke({request.question})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the question: {str(e)}")