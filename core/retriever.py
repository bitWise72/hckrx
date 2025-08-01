# In core/retriever.py
# --------------------
import os
from typing import List
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv, find_dotenv

# --- Load environment variables FIRST (Robust Method) ---
# find_dotenv() will search for the .env file in the parent directories,
# ensuring it's found regardless of where the script is run from.
load_dotenv(find_dotenv())

# --- Configuration ---
# Now, when this line runs, os.environ.get() will find the key.
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Configure the Gemini API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# --- Constants ---
INDEX_NAME = "hackrx-rag-index"
EMBEDDING_MODEL = "models/embedding-001"
VECTOR_DIMENSION = 768

def embed_and_store(chunks: List[str], document_id: str):
    """
    Embeds text chunks using the Gemini API and stores them in a Pinecone index.
    """
    print(f"Embedding {len(chunks)} chunks for document: {document_id}")

    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)

    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings = result['embedding']
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        raise

    vectors_to_upsert = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{document_id}-{i}"
        metadata = {"text": chunk, "document_id": document_id}
        vectors_to_upsert.append((vector_id, embedding, metadata))

    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)

    print("Successfully stored embeddings in Pinecone.")

def retrieve_context(query: str, document_id: str) -> str:
    """
    Embeds a query and retrieves the most relevant text chunks from Pinecone.
    """
    index = pc.Index(INDEX_NAME)

    try:
        query_result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = query_result['embedding']
    except Exception as e:
        print(f"Error during query embedding: {e}")
        raise

    query_response = index.query(
        vector=query_embedding,
        top_k=4,
        include_metadata=True,
        filter={"document_id": document_id}
    )

    context = "\n---\n".join([match['metadata']['text'] for match in query_response['matches']])
    return context
