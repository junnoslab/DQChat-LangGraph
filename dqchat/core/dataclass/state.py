from collections.abc import Iterator
from typing import Optional

from datasets import Dataset, IterableDataset
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from transformers import PreTrainedModel
from transformers.pipelines import Pipeline
from vllm import LLM, SamplingParams


class QAState(BaseModel):
    question: str = Field(default="")


class DatasetGeneratorState(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    questions: Optional[IterableDataset] = Field(default=None)
    response_invoker: Optional[Iterator] = Field(default=None)
    responses: Optional[Dataset] = Field(default=None)


class State(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    llm: Optional[PreTrainedModel | LLM | Pipeline] = Field(default=None)
    """LLM model for generating responses"""
    retriever: Optional[BaseRetriever] = Field(default=None)
    """Retriever for finding relevant documents"""
    sampling_params: Optional[SamplingParams] = Field(default=None)
    """Sampling parameters for generating responses with vLLM"""

    question_answer: QAState = Field(default=QAState())
    dataset_generator: DatasetGeneratorState = Field(default=DatasetGeneratorState())
