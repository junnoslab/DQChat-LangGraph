from typing import Union

from datasets import Dataset
from transformers import PreTrainedModel
from transformers.pipelines import Pipeline

from .. import BaseMetric, LLMTestCase


class FaithfulnessMetric(BaseMetric):
    def __init__(self, threshold: float):
        super().__init__(threshold)

    def measure(self, test_case: LLMTestCase) -> float:
        pass

    def evaluate(
        self, dataset: Dataset, llm: Union[PreTrainedModel, Pipeline]
    ) -> Dataset:
        pass
