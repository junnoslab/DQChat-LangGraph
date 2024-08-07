from typing import Optional, TypedDict

from datasets import IterableDataset
from langchain_core.retrievers import BaseRetriever

from ..const import (
    HF_DATASET_NAME,
    HF_CONFIG_QUESTIONS,
    HF_QUESTIONS_DATASET_CONTENT_COL,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    HF_LLM_MODEL_NAME,
    HF_EMBEDDING_MODEL_NAME,
)
from ..data.parser import RAFTResponse
from ..secret import HF_ACCESS_TOKEN


class State(TypedDict):
    questions: Optional[IterableDataset]
    retriever: Optional[BaseRetriever]
    responses: list[RAFTResponse]


class Config(TypedDict):
    model_name: str
    embedding_model_name: str
    dataset_name: str
    dataset_config: str
    dataset_questions_content_column: str
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
            dataset_questions_content_column=HF_QUESTIONS_DATASET_CONTENT_COL,
            hf_access_token=HF_ACCESS_TOKEN,
            vector_db_path=CHROMA_DB_PATH,
            vector_db_collection_name=CHROMA_COLLECTION_NAME,
        )
