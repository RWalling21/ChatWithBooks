from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.chat_models import ChatGooglePalm

page_search = DuckDuckGoSearchRun()

# Allows the AI to search the web to reference a given book
class browseBooks(BaseTool):
    name = "Web Search"
    description = "useful for when you don't know the answer to a question, or need to reference a recently published book"

    def _run(self, query: str) -> str:
        results = self.page_search.search(query)

        return results

# Allows the AI to think through a difficult question 
class chainOfThough(BaseTool):
    name = "Chain of Thought"
    description = "useful for when you are posed a difficult question, and need to logically think it through"

    def __init__(self):
        llm = ChatGooglePalm()

    def _run(self, question: str):
        # Find the key components of the question being asked 
        key_components = self.identify_components(question)
        # break down each component from the key components 
        thoughts = [self.process_component(component) for component in key_components]
        # Using this newly gained insight, answer the question 
        final_answer = self.synthesize_thoughts(thoughts)
        return final_answer

    def identify_components(self, question: str):
        # Use LLM to break down the question
        prompt = f"""Break down the following question into key components: {question}
        You output MUST separate each key component with a '\n' character."""

        response = self.llm.generate(prompt)

        return response.split('\n')

    def process_component(self, component: str):
        # Use LLM to process the component
        prompt = f"Explain the following component of the question: {component}"

        return self.llm.generate(prompt)

    def synthesize_thoughts(self, thoughts: list):
        # Use LLM to synthesize thoughts
        prompt = "Combine the following thoughts into a comprehensive answer:\n" + '\n'.join(thoughts)
        return self.llm.generate(prompt)

tools = [browseBooks, chainOfThough]