# Chatbot API with Retrieval-Augmented Generation (RAG)

This repository contains the code for a FastAPI-powered chatbot API built using the Retrieval-Augmented Generation (RAG) framework. The API enables users to ask questions and receive responses based on data retrieved from uploaded documents.

*Features*

    Dynamic Document Retrieval: Extracts relevant context from PDFs to answer user queries.
    Interactive Endpoints: Endpoints for fetching document content, asking questions, and more.
    Scalable Architecture: Easily extensible with features like additional data uploads or user authentication.

Medium Article

This code is part of my Medium article where I explain the concepts and implementation in detail. Check out the article for a step-by-step guide on building and deploying this API.
How to Use

    Clone the Repository

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

Install Dependencies

pip install -r requirements.txt

Run the API

    uvicorn main:app --host 0.0.0.0 --port 8000

    Endpoints
        GET /: Check if the API is running.
        GET /page/{index}: Retrieve specific document content.
        POST /ask: Ask a question based on the documents.

Enhancements

This API can be further enhanced by:

    Adding a route to upload new documents dynamically.
    Implementing user authorization for secure access.

Deployment

The API is containerized using Docker and can be deployed to platforms like Google Cloud Run for scalable hosting.
License

This project is licensed under the MIT License.
