from .. import LLMTestCase
from ..faithfulness import FaithfulnessMetric


class TestFaithfulnessMetric:
    def test_measure(self):
        metric = FaithfulnessMetric(threshold=0.5)
        test_case = LLMTestCase(
            input="Hello, world!",
            actual_output="Hello, world!",
            expected_output="Hello, world!",
        )
        assert metric.measure(test_case) is None
