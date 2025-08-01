# In api/routes.py
# ------------------
from fastapi import APIRouter, Body, HTTPException
from .schemas import HackRxRequest, HackRxResponse
from core.parser import load_and_chunk_document
from core.retriever import embed_and_store, retrieve_context
from core.generator import get_answer_from_llm
import hashlib
import time

# Create an API router
# This is the line that creates the 'router' object that main.py needs to import.
router = APIRouter()

@router.post("/hackrx/run", response_model=HackRxResponse)
async def run_submission(request: HackRxRequest = Body(...)):
    """
    This is the main endpoint that processes a document and answers questions.
    """
    start_time = time.time()
    try:
        # Use a hash of the URL as a unique ID for the document.
        # This helps in filtering results in the vector database.
        document_url = str(request.documents)
        document_id = hashlib.sha256(document_url.encode()).hexdigest()

        # STEP 1: Parse and Chunk the document
        chunks = load_and_chunk_document(document_url)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to extract text from the document.")

        # STEP 2: Embed chunks and store them in Pinecone
        # This step can be slow the first time a document is processed.
        embed_and_store(chunks, document_id)

        # STEP 3: Process each question to generate answers
        final_answers = []
        for question in request.questions:
            print(f"Processing question: '{question}'")

            # 3a. Retrieve relevant context from Pinecone
            context = retrieve_context(question, document_id)

            # 3b. Generate a final answer using the LLM
            answer = get_answer_from_llm(question, context)
            final_answers.append(answer)

        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds.")
        return HackRxResponse(answers=final_answers)

    except Exception as e:
        # Catch any exceptions and return a generic server error.
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
