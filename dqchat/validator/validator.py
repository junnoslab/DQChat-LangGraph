from typing import Literal
import logging
import re

from datasets import Dataset
from langchain_huggingface import HuggingFacePipeline
from ragas import RunConfig, metrics, adapt, evaluate
from sentence_transformers import SentenceTransformer
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
        pipe = guard_type(state.llm, Pipeline)
        embedding_model = guard_type(state.embedding_model, SentenceTransformer)
    except TypeError as e:
        _LOGGER.error(f"Failed to validate state: {e}")
        return "invalid"

    # Split content of `context` by separator " " to list of strings.
    dataset = dataset.map(
        lambda x: {"contexts": re.split(r"(?<=[.!?])\s+", x["context"])}
    )
    dataset = dataset.remove_columns(["context"])
    _LOGGER.info(dataset[0])

    pipe = HuggingFacePipeline(
        pipeline=pipe,
        pipeline_kwargs={
            "max_new_tokens": 1024 * 4,
            "eos_token_id": pipe.tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
        },
        model_kwargs={
            "max_length": 1024 * 8,
        },
        batch_size=16,
    )

    adapt(
        metrics=[metrics.faithfulness, metrics.answer_relevancy],
        language="ko",
        llm=pipe,
    )

    score = evaluate(
        dataset=dataset,
        metrics=[metrics.faithfulness, metrics.answer_relevancy],
        llm=pipe,
        embeddings=embedding_model,
        run_config=RunConfig(
            timeout=999_999_999,
        ),
    )
    df: pd.DataFrame = score.to_pandas()
    _LOGGER.info(df.head(20))

    return "valid"
