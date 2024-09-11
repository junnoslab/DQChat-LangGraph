import torch
from transformers import pipeline

from dqchat.validator.metrics import LLMTestCase
from dqchat.validator.metrics.faithfulness import FaithfulnessMetric


class TestFaithfulnessMetric:
    metric = FaithfulnessMetric(threshold=0.2)
    pipe = pipeline(
        "text-generation",
        "MLP-KTLim/llama-3-Korean-Bllossom-8B",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    mock_case = LLMTestCase(
        input="아인슈타인은 누구인가요?",
        actual_output="아인슈타인은 1968년에 노벨상을 수상한 독일 과학자입니다.",
        contexts=[
            "아인슈타인은 광전 효과 발견으로 노벨상을 수상했습니다.",
            "아인슈타인은 1968년에 노벨상을 수상했습니다.",
            "아인슈타인은 독일 과학자입니다.",
        ],
    )

    def test_generate_truths(self):
        self.metric.pipeline = self.pipe
        chain = self.metric._generate_truths()
        truths = chain.invoke(self.mock_case.to_dict())
        print(truths)

    # def test_measure(self):
    #     assert self.metric.evaluate(dataset=self.mock_case, llm=self.pipe) >= 0.2
