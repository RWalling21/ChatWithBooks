from marvin import ai_model
from pydantic import BaseModel, Field

# Define Marvin AI Text classifier
@ai_model
class BookQuery(BaseModel):
    books: str = Field(..., description="The book being referenced")
    query: str = Field(..., description="The question being asked about the book")

# Extract the book and question from the given query
def extract_book_and_question(input_text):
    # Process the text with marvin
    response = BookQuery(input_text)
    return {"book": response.books, "query": response.query}