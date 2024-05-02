from langchain import hub
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from src.helper import get_llm


class QAChain:
    """
    Represents a question-answering chain that combines a retriever and a language model.

    Args:
        retriever: The retriever object used to retrieve relevant information.
        llm (str): The language model used for generating responses.

    Attributes:
        llm: The language model used for generating responses.
        prompt: The prompt object for the language model.
        retriever: The retriever object used to retrieve relevant information.
    """

    def __init__(self, retriever, llm: str):
        self.llm = get_llm(llm)
        self.prompt = hub.pull("rlm/rag-prompt")
        self.retriever = retriever

    def get_langchain_qa_chain(self):
        """
        Returns the question-answering chain for the language model.
        """
        return (
            {
                "context": itemgetter("question") | self.retriever,
                "question": itemgetter("question"),
            }
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": self.prompt | self.llm, "context": itemgetter("context")}
        )
