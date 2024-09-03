from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch

from ..core import State


def load_model(state: State, config: dict) -> State:
    model_framework = config["configurable"]["model_framework"]

    if model_framework == "huggingface":
        load_pipeline(state=state, config=config)

    return state


def load_pipeline(state: State, config: dict) -> State:
    config = config["configurable"]

    pipe = pipeline(
        task="text-generation",
        model=config.get("model_name", ""),
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "cache_dir": config.get("model_cache_path", None),
        },
        device_map="auto",
    )
    state.llm = pipe

    return state


def load_embedding_model(state: State, config: dict) -> State:
    embedding_model_name = config["configurable"]

    embedding_model = SentenceTransformer(
        embedding_model_name.get("embedding_model_name", ""),
        cache_folder=config.get("model_cache_path", None),
    )
    state.embedding_model = embedding_model

    return state
