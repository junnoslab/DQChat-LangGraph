from typing import List

from chromadb import Collection, PersistentClient, QueryResult
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

from ..core.dataclass.state import State
from ..utils.type_helper import guard_let


class Store:
    collection: Collection
    embedding_model: SentenceTransformer

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        db_path: str = "db",
        collection_name: str = "dqchat",
    ):
        self.embedding_model = embedding_model

        chroma = PersistentClient(path=db_path)
        collection = chroma.get_collection(name=collection_name)
        self.collection = collection

    def query(self, query: str, top_k: int = 10) -> QueryResult:
        # Update query to get batch
        encoded_query = self.embedding_model.encode(query)

        if isinstance(encoded_query, List):
            query_embedding = encoded_query
        elif isinstance(encoded_query, torch.Tensor):
            query_embedding = encoded_query.tolist()
        elif isinstance(encoded_query, np.ndarray):
            query_embedding = encoded_query.tolist()

        results = self.collection.query(
            query_embeddings=query_embedding, n_results=top_k
        )
        return results


class Retriever(BaseRetriever):
    # Using chromadb directly for now since langchain doesn't support retrieving `doc_id` by similarity search.
    vector_store: Store
    top_k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = self.vector_store.query(query=query, top_k=self.top_k)

        def convert_query_result_to_document(
            query_result: QueryResult,
        ) -> list[Document]:
            documents: list[Document] = []

            ids = query_result["ids"][0]
            metadatas = query_result["metadatas"][0]
            distances = query_result["distances"][0]

            for id, metadata, distance in zip(ids, metadatas, distances):
                metadata["distance"] = distance
                document = Document(
                    id=id, page_content=metadata.get("query", ""), metadata=metadata
                )
                documents.append(document)

            return documents

        docs = convert_query_result_to_document(query_result=results)

        return docs


def prepare_retriever(state: State, config: dict) -> State:
    """
    Retrieve the data from the vector store.\n
    :param state: GraphState
    :param config: Configuration for Graph
    :return: State
    """
    config = config["configurable"]

    embedding_model = guard_let(state.embedding_model, SentenceTransformer)

    store = Store(
        embedding_model=embedding_model,
        db_path=config["vector_db_path"],
        collection_name=config["vector_db_collection_name"],
    )
    retriever = Retriever(vector_store=store)

    state.retriever = retriever

    return state
