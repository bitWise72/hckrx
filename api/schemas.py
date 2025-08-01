# In api/schemas.py
# -------------------
from pydantic import BaseModel, HttpUrl
from typing import List

# Pydantic models define the structure of your API requests and responses.
# They provide automatic data validation and documentation.
# This is the class that routes.py is trying to import.
class HackRxRequest(BaseModel):
    documents: HttpUrl  # Ensures the input is a valid URL
    questions: List[str]

# This is the other class that routes.py is trying to import.
class HackRxResponse(BaseModel):
    answers: List[str]
