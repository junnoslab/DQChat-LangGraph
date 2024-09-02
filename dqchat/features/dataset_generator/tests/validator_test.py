from datasets import Dataset, load_dataset
import pytest

from ....core import State, default_config
from ....core.dataclass.state import DatasetGeneratorState
from ....llm.loader import load_pipeline
from ....utils.type_helper import guard_type
from ....utils.secret import HF_ACCESS_TOKEN
from ..validator import validate_dataset


class TestDataset:
    @pytest.fixture(scope="class")
    def state(self) -> State:
        ds = load_dataset(
            path="Junnos/DQChat-raft",
            name="question-answer",
            split="train",
            token=HF_ACCESS_TOKEN,
        )
        dataset = guard_type(ds, Dataset)
        return State(
            dataset_generator=DatasetGeneratorState(
                responses=dataset,
            )
        )

    @pytest.fixture(scope="class")
    def config(self) -> dict:
        return {
            "configurable": default_config(),
        }

    def test_dataset_is_loaded(self, state: State, config: dict):
        assert state.dataset_generator.responses.num_rows > 0

    def test_dataset_is_valid(self, state: State, config: dict):
        load_pipeline(state=state, config=config)
        validation_result = validate_dataset(state=state, config=config)
        assert validation_result == "valid"
