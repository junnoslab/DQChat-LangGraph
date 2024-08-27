from collections.abc import Iterator
from typing import Optional, Union

from datasets import Dataset
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain_core.retrievers import BaseRetriever
from transformers import PreTrainedModel
from transformers.pipelines import Pipeline


class QAState(BaseModel):
    question: str = Field(default="")


class DatasetGeneratorState(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    questions: Optional[Dataset] = Field(default=None)
    response_invoker: Optional[Iterator] = Field(default=None)
    responses: Optional[Dataset] = Field(default=None)

    ds_id: int = Field(default=1)

    @property
    def dataset_id(self) -> str:
        return f"ds{self.ds_id:03d}"

    qa_id: int = Field(default=1)

    @property
    def id(self) -> str:
        return f"qa{self.qa_id:03d}"

    @validator("ds_id", "qa_id", pre=True)
    def id_alphanumeric(cls, v: Union[int, str]) -> int:
        if isinstance(v, int):
            return v
        elif isinstance(v, str) and v[2:].isdigit():
            return int(v[2:])
        else:
            raise ValueError(f"Invalid id format:{v}")


class TrainerState(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    dataset: Optional[Dataset] = Field(default=None)
    invoker: Optional[Iterator] = Field(default=None)


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
    trainer: TrainerState = Field(default=TrainerState())
