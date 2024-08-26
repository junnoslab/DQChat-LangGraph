from collections.abc import Iterator
from typing import Optional, Union

from datasets import Dataset, IterableDataset
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from transformers import PreTrainedModel
from transformers.pipelines import Pipeline


class QAState(BaseModel):
    question: str = Field(default="")


class DatasetGeneratorState(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    questions: Optional[IterableDataset] = Field(default=None)
    response_invoker: Optional[Iterator] = Field(default=None)
    responses: Optional[Dataset] = Field(default=None)
    _next_id: int = Field(default=1)

    @property
    def next_id(self) -> str:
        return f"qa{self._next_id:03d}"

    @next_id.setter
    def next_id(self, value: Union[int, str]):
        prefix = "qa"

        if isinstance(value, int):
            self._next_id = value
        elif isinstance(value, str) and value[len(prefix) :].isdigit():
            self._next_id = int(value[len(prefix) :])
        else:
            raise ValueError(f"Invalid id format: {value}")


class State(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    llm: Optional[PreTrainedModel | Pipeline] = Field(default=None)
    """LLM model for generating responses"""
    retriever: Optional[BaseRetriever] = Field(default=None)
    """Retriever for finding relevant documents"""
    # sampling_params: Optional[SamplingParams] = Field(default=None)
    # """Sampling parameters for generating responses with vLLM"""

    question_answer: QAState = Field(default=QAState())
    dataset_generator: DatasetGeneratorState = Field(default=DatasetGeneratorState())
