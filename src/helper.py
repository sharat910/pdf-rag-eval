from langchain_openai import ChatOpenAI
from langchain_community.llms.ollama import Ollama


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_llm(llm: str):
    if llm == "gpt-3.5-turbo":
        return ChatOpenAI(model="gpt-3.5-turbo")
    elif llm == "ollama":
        return Ollama(model="llama3")
    else:
        raise ValueError(f"Unknown LLM: {llm}")
