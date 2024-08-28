from collections.abc import Iterator
import logging

from datasets import Dataset
from langchain_core.prompts import PromptTemplate
from langchain_core.prompt_values import StringPromptValue
from langchain_core.retrievers import BaseRetriever
from transformers import PreTrainedTokenizerBase
from transformers.pipelines import Pipeline
import tqdm

from . import PROMPT_TEMPLATE
from .output_parser import ParserError, QAResponse, QAResponseParser
from .retriever_parser import RetrieverParser
from ..feature import BaseFeature
from ...core import State
from ...utils.type_helper import guard_type


_LOGGER = logging.getLogger(__file__)


def prepare_invoker(state: State, config: dict) -> State:
    qa_agent = DatasetBuilder(state=state, config=config)

    invoker = qa_agent.invoker
    state.dataset_generator.response_invoker = invoker

    return state


def invoke(state: State, config: dict) -> State:
    invoker = state.dataset_generator.response_invoker

    if invoker is None:
        raise ValueError("No invoker to run.")

    for response in invoker:
        qa_response = guard_type(response, QAResponse).dict()

        # Save in memory
        try:
            dataset = guard_type(state.dataset_generator.responses, Dataset)
            dataset = dataset.add_item(qa_response)
        except TypeError:
            dataset = Dataset.from_list([qa_response])

        state.dataset_generator.responses = dataset

    return state


class DatasetBuilder(BaseFeature[QAResponse]):
    retriever: BaseRetriever
    pipe: Pipeline
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, state: State, config: dict):
        super().__init__(state, config)
        self.retriever = guard_type(state.retriever, BaseRetriever)
        self.pipe = guard_type(state.llm, Pipeline)
        self.tokenizer = guard_type(self.pipe.tokenizer, PreTrainedTokenizerBase)

    def __prompt_iterator(self, dataset: Dataset) -> Iterator[str]:
        for question_dict in dataset:
            question = question_dict["question"]

            prompt_builder: Runnable = {
                "context": self.retriever | RetrieverParser.docs_to_context,
                "question": RunnablePassthrough(),
            } | PromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt_template = prompt_builder.invoke(input=question)
            prompt = guard_type(prompt_template, StringPromptValue)
            yield prompt.to_string()

    def build_invoker(self) -> Iterator[QAResponse]:
        questions = guard_type(self.state.dataset_generator.questions, Dataset)

        # Set tokenizer's padding side to left since it's decoder-only model.
        self.tokenizer.padding_side = "left"

        def invoke_iterator(dataset: Dataset) -> Iterator[QAResponse]:
            # TODO: Extract magic numbers to config
            pipeline_iterator = self.pipe(
                self.__prompt_iterator(dataset=dataset),
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                eos_token_id=[self.tokenizer.eos_token_id],
                batch_size=16,
            )

            for output in tqdm.tqdm(pipeline_iterator, total=len(dataset)):
                list_output = guard_type(output, list)

                generated_text = list_output[0]["generated_text"]
                text = guard_type(generated_text, str)

                parser = QAResponseParser(state=self.state, config=self.config)
                response = parser.parse(text)
                # Increase qa_id after parsing
                self.state.dataset_generator.qa_id += 1

                try:
                    safe_response = guard_type(response, QAResponse)
                    yield safe_response
                except ParserError as e:
                    _LOGGER.warning(e)
                    continue

        iterator = invoke_iterator(dataset=questions)

        return iterator
