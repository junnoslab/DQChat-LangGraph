from collections.abc import Iterator
from typing import Any, TypedDict

from datasets import IterableDataset
from langchain_core.prompts import PromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from transformers import AutoTokenizer
from transformers.pipelines import Pipeline
import tqdm

from . import PROMPT_TEMPLATE
from .output_parser import QAResponse, RAFTResponseParser
from .retriever_parser import RetrieverParser
from ..feature import BaseFeature
from ...core import State
from ...utils.type_helper import guard_type


def dataset_invoker_chain_builder(state: State, config: dict) -> State:
    qa_agent = DatasetBuilder(state=state, config=config)

    invoker = qa_agent.invoker
    state.dataset_generator.response_invoker = invoker

    return state


def dataset_invoker(state: State, config: dict) -> State:
    invoker = state.dataset_generator.response_invoker

    if invoker is None:
        raise ValueError("No invoker to run.")

    for response in invoker:
        print(response)

    return state


class PipelineInput(TypedDict):
    pipeline: Pipeline
    prompt: ChatPromptValue
    config: dict[str, Any]


class DatasetBuilder(BaseFeature[QAResponse]):
    def build_chain(self) -> Runnable:
        config = self.config["configurable"]
        prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)

        # Prepare components
        retriever = guard_type(self.state.retriever, BaseRetriever)
        pipe = guard_type(self.state.llm, Pipeline)

        if retriever is None:
            raise ValueError("Retriever is not initialized.")

        def __run_pipeline(inputs: PipelineInput) -> str:
            pipeline: Pipeline = inputs["pipeline"]
            prompt: ChatPromptValue = inputs["prompt"]
            pipeline_configuration: dict[str, Any] = inputs["config"]

            llm_model_name: str = pipeline_configuration["model_name"]
            tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

            output = pipeline(
                prompt.to_string(),
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                eos_token_id=[tokenizer.eos_token_id],
            )
            list_output = guard_type(output, list)

            return list_output[0]["generated_text"]

        chain: Runnable = (
            {
                "context": retriever | RetrieverParser.docs_to_context,
                "question": RunnablePassthrough(),
            }
            | prompt_template
            | {
                "prompt": RunnablePassthrough(),
                "pipeline": lambda _: pipe,
                "config": lambda _: config,
            }
            | RunnableLambda(__run_pipeline)
            | RAFTResponseParser()
        )

        return chain

    def build_invoker(self) -> Iterator[QAResponse]:
        questions = self.state.dataset_generator.questions

        if questions is None:
            raise ValueError("Questions are not initialized.")

        rag_chain: Runnable = self.build_chain()

        def rag_invoke_iterator(chain: Runnable, dataset: IterableDataset):
            pbar = tqdm.tqdm(dataset, total=3480)
            for question_dict in pbar:
                pbar.set_description(f"{question_dict['category']}")
                pbar.set_postfix(q=question_dict["question"])
                question = question_dict["question"]
                output = chain.invoke(input=question)

                response_output = guard_type(output, QAResponse)

                yield response_output

        iterator = rag_invoke_iterator(rag_chain, questions)

        return iterator
