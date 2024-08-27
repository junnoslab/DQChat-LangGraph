from collections.abc import Iterator

from datasets import Dataset
from langchain_core.prompts import PromptTemplate
from langchain_core.prompt_values import StringPromptValue
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from transformers.pipelines import Pipeline
import tqdm

from . import PROMPT_TEMPLATE
from .output_parser import QAResponse, QAResponseParser
from .retriever_parser import RetrieverParser
from ..feature import BaseFeature
from ...core import State
from ...utils.type_helper import guard_type


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
    def build_invoker(self) -> Iterator[QAResponse]:
        questions = self.state.dataset_generator.questions
        questions_dataset = guard_type(questions, Dataset)

        retriever = guard_type(self.state.retriever, BaseRetriever)
        pipe = guard_type(self.state.llm, Pipeline)
        tokenizer = pipe.tokenizer
        tokenizer.padding_side = "left"

        def invoke_iterator(dataset: Dataset) -> Iterator[QAResponse]:
            def prompt_iterator() -> Iterator[str]:
                for question_dict in dataset:
                    question = question_dict["question"]

                    prompt_builder: Runnable = (
                        {
                            "context": retriever | RetrieverParser.docs_to_context,
                            "question": RunnablePassthrough(),
                        }
                        | PromptTemplate.from_template(PROMPT_TEMPLATE)
                    )
                    prompt_template = prompt_builder.invoke(input=question)
                    prompt = guard_type(prompt_template, StringPromptValue)
                    yield prompt.to_string()

            pipeline_iterator = pipe(
                prompt_iterator(),
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                eos_token_id=[tokenizer.eos_token_id],
                batch_size=32,
            )
            for output in tqdm.tqdm(pipeline_iterator):
                list_output = guard_type(output, list)

                generated_text = list_output[0]["generated_text"]
                text = guard_type(generated_text, str)

                parser = QAResponseParser(state=self.state, config=self.config)
                response = parser.parse(text)
                # Update qa_id after parsing
                self.state.dataset_generator.qa_id += 1

                try:
                    safe_response = guard_type(response, QAResponse)
                    yield safe_response
                except TypeError:
                    print("Skipping due to JSON parsing error.")
                    continue

        iterator = invoke_iterator(dataset=questions_dataset)

        return iterator
