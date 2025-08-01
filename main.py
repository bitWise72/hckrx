# In main.py
# ------------
from fastapi import FastAPI
from api.routes import router as api_router
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
# This line looks for a .env file in the same directory and loads its content.
load_dotenv()

# Check for essential environment variables on startup
# This is a safety check to make sure your API keys are loaded.
if not all([os.getenv("GOOGLE_API_KEY"), os.getenv("PINECONE_API_KEY"), os.getenv("PINECONE_ENVIRONMENT")]):
    # If any key is missing, the program will stop with this error.
    # Make sure your .env file is filled out correctly.
    raise RuntimeError("Missing essential environment variables. Please check your .env file.")

# Create the FastAPI application instance
# This is the line Uvicorn is looking for. The variable MUST be named 'app'.
app = FastAPI(
    title="Intelligent Query–Retrieval System",
    description="An API to answer questions about documents using RAG with Gemini and Pinecone.",
    version="1.0.0"
)

# Include the API router from the api/routes.py file
app.include_router(api_router, prefix="/api/v1")

@app.get("/", tags=["Root"])
def read_root():
    """
    A simple root endpoint to confirm the API is running.
    You can see this by going to http://127.0.0.1:8000 in your browser.
    """
    return {"message": "Welcome to the Query-Retrieval System API. Head to /docs for API documentation."}
