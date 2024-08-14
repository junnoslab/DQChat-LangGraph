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
from ..utils.runmode import RunMode


class State(TypedDict):
    questions: Optional[IterableDataset]
    retriever: Optional[BaseRetriever]
    responses: list[RAFTResponse]


class Config(TypedDict):
    run_mode: RunMode
    model_name: str
    embedding_model_name: str
    dataset_name: str
    dataset_config: str
    dataset_questions_content_column: str
    hf_access_token: str

    vector_db_path: str
    vector_db_collection_name: str

    @classmethod
    def default_config(
        cls,
        run_mode: Optional[RunMode] = None,
        model_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config: Optional[str] = None,
        dataset_questions_content_column: Optional[str] = None,
        hf_token: Optional[str] = None,
        vector_db_path: Optional[str] = None,
        vector_db_collection_name: Optional[str] = None,
    ) -> "Config":
        return Config(
            run_mode=run_mode or "inference",
            model_name=model_name or HF_LLM_MODEL_NAME,
            embedding_model_name=embedding_model_name or HF_EMBEDDING_MODEL_NAME,
            dataset_name=dataset_name or HF_DATASET_NAME,
            dataset_config=dataset_config or HF_CONFIG_QUESTIONS,
            dataset_questions_content_column=dataset_questions_content_column
            or HF_QUESTIONS_DATASET_CONTENT_COL,
            hf_access_token=hf_token or HF_ACCESS_TOKEN,
            vector_db_path=vector_db_path or CHROMA_DB_PATH,
            vector_db_collection_name=vector_db_collection_name
            or CHROMA_COLLECTION_NAME,
        )
