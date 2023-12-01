from langchain.tools import BaseTool, DuckDuckGoSearchRun

# Allows the AI to search the web to reference a given book
class browseBooks(BaseTool):
    name = "Web Search"
    description = "useful for when you don't know the answer to a question, or need to reference a recently published book"

    def _run(self, query: str) -> str:
        page_search = DuckDuckGoSearchRun()
        results = page_search.run(query)

        return str(results)