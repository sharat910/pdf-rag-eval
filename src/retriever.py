import os
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from typing import List


class Retriever:

    def __init__(self, files: List[str], retriever_mode: str, embedding: str, **kwargs):
        """
        Initialize the Retriever object.

        Args:
            files (List[str]): List of file paths to load.
            retriever_mode (str): Mode of the retriever.
            embedding (str): Type of embedding to use.
            **kwargs: Additional keyword arguments.

        """
        self._setup_api_keys(embedding)
        self.raw_docs = self._load_files(files)
        self.parent_splitter = MarkdownTextSplitter(
            chunk_size=kwargs.get("parent_chunk_size", 1200)
        )
        self.child_splitter = MarkdownTextSplitter(
            chunk_size=kwargs.get("child_chunk_size", 400)
        )
        self.retriever = self._get_retriever(retriever_mode, embedding)

    def get_langchain_retriever(self):
        return self.retriever

    def _setup_api_keys(self, embedding):
        """
        Sets up the API keys based on the specified embedding.

        Args:
            embedding (str): The embedding type ("openai" or "cohere").

        Raises:
            ValueError: If the API key for the specified embedding is not provided.

        """
        if embedding == "openai":
            k = os.getenv("OPENAI_API_KEY")
            if k is None:
                raise ValueError("Please provide your OpenAI API key")
            self.openai_api_key = k
        elif embedding == "cohere":
            k = os.getenv("COHERE_API_KEY")
            if k is None:
                raise ValueError("Please provide your Cohere API key")
            self.cohere_api_key = k

    def _get_vector_store(self, embedding: str):
        """
        Get the vector store for the specified embedding.

        Args:
            embedding (str): The name of the embedding.

        Returns:
            Chroma: The vector store object.

        """
        embedding_func = self._get_embedding(embedding)
        persist_directory = f"./files/vector_store/{embedding}"
        os.makedirs(persist_directory, exist_ok=True)
        return Chroma(
            collection_name=f"allianz_nrma_parent_child__{embedding}",
            embedding_function=embedding_func,
            persist_directory=persist_directory,
        )

    def _get_embedding(self, embedding):
        """
        Returns the embedding object based on the specified embedding type.

        Args:
            embedding (str): The type of embedding to use. Valid options are "openai" and "cohere".

        Returns:
            An instance of the embedding object based on the specified type.

        Raises:
            ValueError: If an invalid embedding type is provided.
        """
        if embedding == "openai":
            return OpenAIEmbeddings(api_key=self.openai_api_key)
        elif embedding == "cohere":
            return CohereEmbeddings(
                model="embed-english-light-v3.0", cohere_api_key=self.cohere_api_key
            )
        else:
            raise ValueError("Invalid embedding")

    def _load_files(self, files):
        """
        Load and process multiple files.

        Args:
            files (list): A list of file paths.

        Returns:
            list: A list of processed documents.
        """
        docs = []
        for file in files:
            loader = TextLoader(file)
            docs.extend(loader.load())
        return docs

    def _get_retriever(self, retriever_mode, embedding):
        """
        Returns the retriever based on the specified mode.

        Args:
            retriever_mode (str): The mode of the retriever. Possible values are "vector", "bm25", or "ensemble".
            embedding: The embedding to be used for vector-based retrievers.

        Returns:
            Retriever: The retriever object based on the specified mode.

        Raises:
            ValueError: If an invalid retriever mode is provided.
        """
        if retriever_mode == "vector":
            self.vector_store = self._get_vector_store(embedding)
            return self._get_vector_retriever()
        elif retriever_mode == "bm25":
            return self._get_bm25_retriever()
        elif retriever_mode == "ensemble":
            self.vector_store = self._get_vector_store(embedding)
            return self._get_ensemble_retriever()
        else:
            raise ValueError("Invalid retriever mode")

    def _get_vector_retriever(self):
        """
        Returns an instance of ParentDocumentRetriever initialized with the necessary parameters.

        Returns:
            ParentDocumentRetriever: An instance of ParentDocumentRetriever.
        """
        retriever = ParentDocumentRetriever(
            parent_splitter=self.parent_splitter,
            child_splitter=self.child_splitter,
            docstore=InMemoryStore(),
            vectorstore=self.vector_store,
        )
        retriever.add_documents(self.raw_docs)
        return retriever

    def _get_bm25_retriever(self):
        """
        Returns a BM25Retriever object with customized parameters.

        This method splits the raw documents using the parent_splitter and creates a BM25Retriever object
        from the parent documents. It then sets the value of `k` to 5 and returns the retriever.

        Returns:
            BM25Retriever: A BM25Retriever object with customized parameters.
        """
        parent_docs = self.parent_splitter.split_documents(self.raw_docs)
        retriever = BM25Retriever.from_documents(parent_docs)
        retriever.k = 5
        return retriever

    def _get_ensemble_retriever(self):
        """
        Returns an instance of EnsembleRetriever that combines the results from vector_retriever and bm25_retriever.

        :return: An instance of EnsembleRetriever.
        """
        vector_retriever = self._get_vector_retriever()
        bm25_retriever = self._get_bm25_retriever()
        return EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever], weights=[0.6, 0.4]
        )
