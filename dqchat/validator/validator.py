from typing import Literal
import logging

from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from ragas import metrics, evaluate
from transformers.pipelines import Pipeline
import pandas as pd

from ..core import State
from ..utils.type_helper import guard_type


_LOGGER = logging.getLogger(__file__)


def validate(state: State, config: dict) -> Literal["valid", "invalid"]:
    """
    Validate the state
    :param state: The state to validate
    :return: "valid" if the state is valid, "invalid" otherwise
    """
    try:
        dataset = guard_type(state.dataset_generator.responses, Dataset)
    except TypeError as e:
        _LOGGER.error(f"Failed to validate state: {e}")
        return "invalid"

    try:
        pipe = guard_type(state.llm, Pipeline)
    except TypeError as e:
        _LOGGER.error(f"Failed to validate state: {e}")
        return "invalid"

    # Split content of `context` by separator " " to list of strings.
    dataset = dataset.map(lambda x: {"contexts": x["context"].split(" ")})

    embedding_model_name = guard_type(
        config["configurable"]["embedding_model_name"], str
    )

    score = evaluate(
        dataset=dataset,
        metrics=[metrics.faithfulness],
        llm=HuggingFacePipeline(pipe),
        embeddings=HuggingFaceEmbeddings(model_name=embedding_model_name),
    )
    df: pd.DataFrame = score.to_pandas()
    _LOGGER.info(df.head(20))
    return "valid"
