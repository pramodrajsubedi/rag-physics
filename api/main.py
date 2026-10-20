import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException # FastAPI is a modern web framework for building APIs with Python.
                                            # HTTPException is used to handle errors in API endpoints.
from pydantic import BaseModel #It's for data validation and serialization.
from rag import ask # Importing the ask function from the rag module, which handles the RAG pipeline.

app = FastAPI(title = "Space Physics RAG API", version = "1.0.0") # Creating an instance of the FastAPI application with a title and version.

class QueryInput(BaseModel): # Defining a Pydantic model for the input data structure.
    question: str # The model expects a single field 'question' of type string.
    top_k: int = 5 # The model also has an optional field 'top_k' of type integer with a default value of 5.

class QueryOutput (BaseModel): # Defining a Pydantic model for the output data structure.
    question: str # The model will return the original question string.
    answer: str # The model will return the generated answer string.
    sources: list[str] # The model will return a list of source titles as strings.

@app.get("/")
def root():
    return {"message" : "Space Physics RAG APi is running"} #This is a simple endpoint that returns a message indicating that the API is running. It can be used for health checks or to verify that the API is accessible.

@app.post("/ask", response_model=QueryOutput)# This endpoint handles POST requests to the "/ask" URL. It expects input data that matches the QueryInput model and will return data that matches the QueryOutput model.
def ask_question(data: QueryInput):# The function takes a QueryInput object as input, which contains the question and the number of top results to retrieve.
    if not data.question.strip(): # Check if the question is empty or contains only whitespace.
        raise HTTPException(status_code=422, detail="Question cannot be empty") # If the question is empty, raise an HTTPException with a 422 status code and a message indicating that the question cannot be empty.
    try:
        result = ask(data.question, top_k=data.top_k) # Call the ask function from the rag module, passing the question and top_k value. This function handles the RAG pipeline: retrieving relevant chunks, building a prompt, and generating an answer.
        if result is None: # If the result from the ask function is None, raise an HTTPException with a 500 status code and a message indicating that the RAG pipeline returned None.
            raise HTTPException(status_code=500, detail="RAG pipeline returned None") # If the RAG pipeline returns None, raise an HTTPException with a 500 status code and a message indicating that the RAG pipeline returned None.
        return QueryOutput(
            question=data.question, # Return the original question from the input data.
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as e: # If any exception occurs during the execution of the ask function, catch the exception and raise an HTTPException with a 500 status code and the exception message as the detail. This ensures that any unexpected errors are properly communicated to the client.
        raise HTTPException(status_code=500, detail=str(e))