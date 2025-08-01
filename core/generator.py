# In core/generator.py
# --------------------
import os
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

# --- Load environment variables FIRST (Robust Method) ---
load_dotenv(find_dotenv())

# --- Configure the Gemini API key ---
# This uses the key from your .env file
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    # This will stop the program if the key is not found or invalid on startup
    raise

# --- Model Configuration (Following latest documentation) ---
# Using gemini-1.5-pro-latest as it's the most capable model available for this task.
GENERATION_MODEL_NAME = "gemini-2.5-pro"

# Configuration for the generation process
generation_config = {
  "temperature": 0.2, # Low temperature for factual, less creative answers
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

# Safety settings to block harmful content
safety_settings = [
  {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the Generative Model with the new configuration
try:
    model = genai.GenerativeModel(model_name=GENERATION_MODEL_NAME,
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
except Exception as e:
    print(f"Error initializing Gemini Model: {e}")
    raise

def get_answer_from_llm(query: str, context: str) -> str:
    """
    Generates a final answer using the configured Gemini Pro model based on the
    retrieved context.
    """
    prompt = f"""
    You are an expert AI assistant specialized in analyzing policy documents.
    Your task is to answer the user's question based *only* on the provided context.
    Do not use any external knowledge or make assumptions.
    If the answer is not available in the context, state that clearly and concisely.

    CONTEXT:
    ---
    {context}
    ---

    QUESTION: {query}

    ANSWER:
    """

    try:
        # The model is already initialized, so we just use it here.
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # This will now catch errors during the generation call itself
        print(f"Detailed Gemini API Error during generation: {e}")
        return "Error: Could not generate an answer due to an API issue."

