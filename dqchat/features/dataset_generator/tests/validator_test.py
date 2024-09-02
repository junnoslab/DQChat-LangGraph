from datasets import Dataset, load_dataset, DownloadMode
import pytest

from ....utils.type_helper import guard_type
from ....utils.secret import HF_ACCESS_TOKEN
from ..validator import validate


class TestDataset:
    @pytest.fixture(scope="class")
    def dataset(self) -> Dataset:
        ds = load_dataset(
            path="Junnos/DQChat-raft",
            name="question-answer",
            split="train",
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
            token=HF_ACCESS_TOKEN,
        )
        dataset = guard_type(ds, Dataset)
        return dataset

    def test_dataset_is_loaded(self, dataset: Dataset):
        assert dataset.num_rows > 0

    def test_dataset_is_valid(self, dataset: Dataset):
        validation_result = validate(dataset, config={})
        assert validation_result == "pass"
