import os

from datasets import Dataset

from ...core import State
from ...utils.type_helper import guard_type


def save(state: State, config: dict) -> State:
    responses = state.dataset_generator.responses
    responses_dataset = guard_type(responses, Dataset)

    output_dir = os.path.join("output", "dataset_builder")

    responses_dataset.to_json(
        os.path.join(output_dir, "responses.jsonl"), force_ascii=False, indent=4
    )
    responses_dataset.to_json(
        os.path.join(output_dir, "responses.json"),
        lines=False,
        force_ascii=False,
        indent=4,
    )
    responses_dataset.to_parquet(os.path.join(output_dir, "responses.parquet"))
    responses_dataset.to_csv(os.path.join(output_dir, "responses.csv"))

    return state
