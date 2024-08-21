"""
DatasetBuilder.

Use LLM to generate responses for prompts and save them as a dataset.
"""

from .dataset_builder import (
    dataset_invoker_chain_builder as dataset_invoker_chain_builder,
    dataset_invoker as dataset_invoker,
)
from .validator import validate as validate
