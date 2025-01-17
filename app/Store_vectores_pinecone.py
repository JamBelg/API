from dotenv import load_dotenv
import os
import getpass

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environmental variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

# Connect to pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Check index
index_name = "pdf-files"  # change if desired
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    print('Create new index')
else:
    print('index already exists')

    index = pc.Index(index_name)
    # Delete all vectors from the index
    index.delete(delete_all=True)

    # Read and split pdfs
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
    
    # openai embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Store vectores in pinecone
    pinecone = PineconeVectorStore.from_documents(
        docs, embedding=embeddings, index_name=index_name
    )
    print(f"New vectores stored in the index '{index_name}'")

