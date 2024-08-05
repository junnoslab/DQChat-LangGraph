from typing import Annotated, Optional, TypedDict, Iterator

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..data.const import HF_DATASET_NAME, HF_CONFIG_QUESTIONS, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME
from ..model.const import HF_LLM_MODEL_NAME, HF_EMBEDDING_MODEL_NAME
from ..secret import HF_ACCESS_TOKEN


class State(TypedDict):
    questions: Optional[Iterator[Document]]
    retriever: Optional[BaseRetriever]


class Config(TypedDict):
    model_name: str
    embedding_model_name: str
    dataset_name: str
    dataset_config: str
    hf_access_token: str

    vector_db_path: str
    vector_db_collection_name: str

    @classmethod
    def default_config(cls) -> "Config":
        return Config(
            model_name=HF_LLM_MODEL_NAME,
            embedding_model_name=HF_EMBEDDING_MODEL_NAME,
            dataset_name=HF_DATASET_NAME,
            dataset_config=HF_CONFIG_QUESTIONS,
            hf_access_token=HF_ACCESS_TOKEN,
            vector_db_path=CHROMA_DB_PATH,
            vector_db_collection_name=CHROMA_COLLECTION_NAME,
        )
