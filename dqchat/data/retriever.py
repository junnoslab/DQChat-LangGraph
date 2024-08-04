from langchain_core.tools import tool


@tool
def vector_search(query: str) -> str:
    if query == "Hey":
        return "Hello, how are you?"
    else:
        return "Yo, what's up?"
