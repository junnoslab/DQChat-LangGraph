from transformers import pipeline
import torch

from ..core import State


def load_pipeline(state: State, config: dict) -> State:
    config = config["configurable"]

    pipe = pipeline(
        task="text-generation",
        model=config.get("model_name", ""),
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "cache_dir": config.get("model_cache_path", None),
        },
    )
    state.llm = pipe

    return state
