# In core/parser.py
# -------------------
import requests
import pypdf
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def load_and_chunk_document(doc_url: str) -> List[str]:
    """
    Downloads a PDF from a URL, extracts its text, and splits it into
    manageable chunks for embedding.

    Args:
        doc_url: The URL of the PDF document.

    Returns:
        A list of text chunks.
    """
    print(f"Downloading and parsing document from: {doc_url}")

    try:
        response = requests.get(doc_url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading document: {e}")
        raise

    # Read PDF from the in-memory content
    pdf_file = io.BytesIO(response.content)
    pdf_reader = pypdf.PdfReader(pdf_file)

    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text:
        print("Warning: No text extracted from the PDF.")
        return []

    # Use a recursive character text splitter for effective chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # The size of each chunk in characters
        chunk_overlap=150, # The overlap between consecutive chunks
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(f"Document successfully split into {len(chunks)} chunks.")
    return chunks
