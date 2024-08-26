from transformers import pipeline
import torch

from ..core import State


def load_model(state: State, config: dict) -> State:
    model_framework = config["configurable"]["model_framework"]

    if model_framework == "huggingface":
        load_pipeline(state=state, config=config)

    return state


def load_pipeline(state: State, config: dict):
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
