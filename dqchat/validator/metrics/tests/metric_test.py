from dqchat.validator.metrics import LLMTestCase
from dqchat.validator.metrics.faithfulness import FaithfulnessMetric


class TestFaithfulnessMetric:
    def test_measure(self):
        metric = FaithfulnessMetric(threshold=0.5)
        test_case = LLMTestCase(
            input="아인슈타인은 누구인가요?",
            actual_output="아인슈타인은 1968년에 노벨상을 수상한 독일 과학자입니다.",
            contexts=[
                "아인슈타인은 광전 효과 발견으로 노벨상을 수상했습니다.",
                "아인슈타인은 1968년에 노벨상을 수상했습니다.",
                "아인슈타인은 독일 과학자입니다.",
            ],
        )
        assert metric.measure(test_case) >= 0.5
