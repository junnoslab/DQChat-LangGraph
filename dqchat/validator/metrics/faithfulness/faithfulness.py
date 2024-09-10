from typing import Optional, Union

from datasets import Dataset
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from transformers import PreTrainedModel
from transformers.pipelines import Pipeline

from dqchat.utils.type_helper import guard_let
from dqchat.validator.metrics.output_parsers import JSONKeyPathOutputParser
from dqchat.validator.metrics import BaseMetric, LLMTestCase
from dqchat.validator.metrics.faithfulness.template import FaithfulnessTemplate


class FaithfulnessMetric(BaseMetric):
    """
    The faithfulness metric measures the quality of your RAG pipeline's generator
    by evaluating whether the `actual_output` factually aligns
    with the contents of your `retrieval_context`.
    """

    model: Optional[PreTrainedModel] = None

    def __init__(self, threshold: float):
        super().__init__(threshold)

    def _generate_truths(self) -> RunnableSequence:
        """
        Generate truths from given retrieval contexts.
        It actually returns a runnable sequence that can be invoked with key `contexts`.
        """
        prompt = PromptTemplate(
            template=FaithfulnessTemplate.generate_truths(text="{contexts}"),
            input_variables=["contexts"],
        )
        chain = RunnableSequence(
            first=prompt,
            middle=self.model,
            last=JSONKeyPathOutputParser(keypath="truths"),
        )
        return chain

    def _generate_claims(self) -> RunnableSequence:
        """
        Generate claims from given actual outputs.
        It actually returns a runnable sequence that can be invoked with key `actual_output`.
        """
        prompt = PromptTemplate(
            template=FaithfulnessTemplate.generate_claims(text="{actual_output}"),
            input_variables=["actual_output"],
        )
        chain = RunnableSequence(
            first=prompt,
            middle=self.model,
            last=JSONKeyPathOutputParser(keypath="claims"),
        )
        return chain

    def _generate_verdicts(self) -> RunnableSequence:
        """
        Generate verdicts from given claims and truths.
        Both `claims` and `truths` are obtained from former steps.
        `claims` is a list of strings, and `truths` is a string which is flattened from obtained contexts.

        It actually returns a runnable sequence that can be invoked with keys `claims` and `truths`.
        """
        prompt = PromptTemplate(
            template=FaithfulnessTemplate.generate_verdicts(
                claims="{claims}", retrieval_context="{truths}"
            ),
            input_variables=["claims", "truths"],
        )
        # Expected output: list of dicts with 'verdict' and Optional 'reason'
        chain = RunnableSequence(
            first=prompt,
            middle=self.model,
            last=JSONKeyPathOutputParser(keypath="verdicts"),
        )
        return chain

    def _generate_reason(self) -> RunnableSequence:
        prompt = PromptTemplate(
            template=FaithfulnessTemplate.generate_reason(text="{claims}"),
            input_variables=["claims"],
        )
        chain = RunnableSequence(first=prompt, last=self.model)
        return chain

    def _calculate_score(self) -> RunnableSequence:
        prompt = PromptTemplate(
            template=FaithfulnessTemplate.calculate_score(text="{verdicts}"),
            input_variables=["verdicts"],
        )
        chain = RunnableSequence(first=prompt, last=self.model)
        return chain

    def measure(self, dataset: Dataset) -> float:
        # 1. Map dataset to LLMTestCase
        test_case = LLMTestCase(
            input=dataset["question"],
            actual_output=dataset["answer"],
            contexts=dataset["context"],
        )

        # 2. Generate truths, claims, verdicts, and reason
        # reason = self._generate_reason() # Inference

        chain = RunnableSequence(
            {
                "truths": self._generate_truths(),
                "claims": self._generate_claims(),
            }
            | self._generate_verdicts()
            | self._calculate_score()
        )

        result = chain.invoke(test_case)
        score_result = guard_let(result, float)
        return score_result

    def evaluate(
        self, dataset: Dataset, llm: Union[PreTrainedModel, Pipeline]
    ) -> Dataset:
        """
        Evaluate the faithfulness of given dataset.\\
        With given LLM using actual outputs, contexts, and pre-defined evaluation prompts,
        generate evaluation scores for each evaluation prompt.
        """
        # 1. Prepare LLM model
        if isinstance(llm, PreTrainedModel):
            model = llm
        elif isinstance(llm, Pipeline):
            model = llm.model
        else:
            raise TypeError("llm must be either a PreTrainedModel or a Pipeline")

        model.eval()
        self.model = model

        # 2. Calculate evaluation scores
        # scores = []

        # 2. Add column with evaluation scores

        # 3. Return dataset
        return dataset
