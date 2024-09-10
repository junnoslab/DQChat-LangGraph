from typing import Optional, Union

from datasets import Dataset
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import PreTrainedModel
from transformers.pipelines import Pipeline, pipeline

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

    pipeline: Optional[Union[Pipeline, HuggingFacePipeline]] = None

    def __init__(self, threshold: float):
        super().__init__(threshold)

    def _wrap_pipeline(
        self, pipeline: Union[Pipeline, HuggingFacePipeline]
    ) -> HuggingFacePipeline:
        try:
            bare_pipe = guard_let(pipeline, Pipeline)
            return HuggingFacePipeline(pipeline=bare_pipe)
        except TypeError:
            return guard_let(pipeline, HuggingFacePipeline)

    def _generate_truths(self) -> RunnableSequence:
        """
        Generate truths from given retrieval contexts.
        It actually returns a runnable sequence that can be invoked with key `contexts`.
        """
        pipe = self._wrap_pipeline(self.pipeline)

        prompt = PromptTemplate(
            template=FaithfulnessTemplate.generate_truths(text="{contexts}"),
            input_variables=["contexts"],
        )
        chain = RunnableSequence(
            first=prompt,
            middle=pipe,
            last=JSONKeyPathOutputParser(keypath="truths"),
        )
        return chain

    def _generate_claims(self) -> RunnableSequence:
        """
        Generate claims from given actual outputs.
        It actually returns a runnable sequence that can be invoked with key `actual_output`.
        """
        pipe = self._wrap_pipeline(self.pipeline)

        prompt = PromptTemplate(
            template=FaithfulnessTemplate.generate_claims(text="{actual_output}"),
            input_variables=["actual_output"],
        )
        chain = RunnableSequence(
            first=prompt,
            middle=pipe,
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
        pipe = self._wrap_pipeline(self.pipeline)

        prompt = PromptTemplate(
            template=FaithfulnessTemplate.generate_verdicts(
                claims="{claims}", retrieval_context="{truths}"
            ),
            input_variables=["claims", "truths"],
        )
        # Expected output: list of dicts with 'verdict' and Optional 'reason'
        chain = RunnableSequence(
            first=prompt,
            middle=pipe,
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

    def measure(self, test_case: LLMTestCase) -> float:
        # 2. Generate truths, claims, verdicts, and reason
        chain = RunnableSequence(
            {
                "truths": self._generate_truths(),
                "claims": self._generate_claims(),
            }
            | self._generate_verdicts()
            | self._calculate_score()
        )

        result = chain.invoke(test_case.to_dict())
        score_result = guard_let(result, float)
        return score_result

    def evaluate(
        self,
        dataset: Union[Dataset, LLMTestCase],
        llm: Union[PreTrainedModel, Pipeline],
    ) -> float:  # TODO: Dataset에 col 추가하여 return 하는 것으로 변경
        """
        Evaluate the faithfulness of given dataset.\\
        With given LLM using actual outputs, contexts, and pre-defined evaluation prompts,
        generate evaluation scores for each evaluation prompt.
        """
        # 1. Prepare dataset as LLMTestCase format
        if isinstance(dataset, Dataset):
            test_case = LLMTestCase(
                input=dataset["question"],
                actual_output=dataset["answer"],
                contexts=dataset["context"],
            )
        elif isinstance(dataset, LLMTestCase):
            test_case = dataset
        del dataset

        # 2. Prepare LLM model
        if isinstance(llm, PreTrainedModel):
            pipe = pipeline(
                task="text-generation",
                model=llm,
                device_map="auto",
            )
        elif isinstance(llm, Pipeline):
            pipe = llm
        del llm

        pipe.model.eval()
        self.pipeline = pipe

        # 3. Calculate evaluation scores
        # scores = []

        # 4. Add column with evaluation scores

        # 5. Return dataset
        # return dataset
        return self.measure(test_case=test_case)
