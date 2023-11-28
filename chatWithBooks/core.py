from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from marvin import ai_model, settings
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import os

# Load Env variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
settings.openai.api_key = os.getenv("OPENAI_API_KEY")

# Define AI Model
@ai_model
class BookQuery(BaseModel):
    books: str = Field(..., description="The book being referenced")
    query: str = Field(..., description="The question being asked about the book")

# Extract the book and question from the given query
def extract_book_and_question(input_text):
    # Process the text with marvin
    response = BookQuery(input_text)
    return {"book": response.books, "query": response.query}

# Prompting 
ANSWER_PROMPT = """
You are ChatWithBooks, a helpful ai assistant with expert level knowledge of all books. Your mission is to answer questions, summarize events, or simply chat with the user about a given book. 

----------

Given the Book: {book}, answer the following query:

{query}

----------
Your output must be well written and directly answer the given question. 
"""

answer_prompt = PromptTemplate.from_template(ANSWER_PROMPT)

if __name__ == "__main__":
    query = 'In the book 1984 by George Orwell, why does Winston say "If there is hope it lies in the proles"'
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

    chain = extract_book_and_question | answer_prompt | llm

    chain_output = chain.invoke(query)

    print(chain_output)
