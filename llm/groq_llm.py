
from langchain_groq import ChatGroq
import os

def initialize_groq():
    """Initializes and returns a ChatGroq object."""
    groq_api_key = os.getenv('GROQ_API_KEY')
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama-3.1-70b-versatile')
    return llm
