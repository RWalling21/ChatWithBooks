from langchain.chat_models import ChatOpenAI
from langserve import add_routes

from fastapi import FastAPI
from marvin import settings

from dotenv import load_dotenv
import os

# Load Env variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
settings.openai.api_key = os.getenv("OPENAI_API_KEY")

# Prompting 
from prompt import answer_prompt, query_prompt

# Import classifier
from classifiers import extract_book_and_question

# Define langchain LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

# Define Chain
chain = query_prompt | extract_book_and_question | answer_prompt | llm

# Serve webserver
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# Create routes
add_routes(
    app,
    chain,
    path="/chatwithbooks",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
