from collections.abc import Iterator

from datasets import IterableDataset
from langchain.prompts.base import BasePromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
)
from transformers import AutoTokenizer
from transformers.pipelines import Pipeline
import tqdm

from .output_parser import RAFTResponse, RAFTResponseParser
from ..const import RAG_PROMPT_TEMPLATE
from ..feature import BaseFeature
from ...core import State


def dataset_invoker_chain_builder(state: State, config: dict) -> State:
    qa_agent = DatasetBuilder(state=state, config=config)

    state.dataset_generator.response_invoker = qa_agent.invoker

    return state


def dataset_invoker(state: State, config: dict) -> State:
    invoker = state.dataset_generator.response_invoker

    if invoker is None:
        raise ValueError("No invoker to run.")

    for response in invoker:
        print(response)

    return state


class DatasetBuilder(BaseFeature[RAFTResponse]):
    def __run_pipeline(self, inputs: dict) -> str:
        pipeline: Pipeline = inputs["pipeline"]
        prompt: ChatPromptValue = inputs["prompt"]
        config = inputs["config"]

        llm_model_name: str = config.get("model_name", "")
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

        output = pipeline(
            prompt.to_string(),
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            eos_token_id=[tokenizer.eos_token_id],
        )

        if not isinstance(output, list):
            raise ValueError("Pipeline output is not a list.")

        return output[0]["generated_text"]

    def build_prompt(self) -> BasePromptTemplate:
        return ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=["context"], template=RAG_PROMPT_TEMPLATE
                    )
                ),
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=["question"],
                        template="{question}<|eot_id|>\n\nAssistant:",
                    )
                ),
            ],
            input_variables=["context", "question"],
        )

    def build_chain(self) -> Runnable:
        prompt_template = self.build_prompt()

        if isinstance(self.state.llm, Pipeline):
            pipe = self.state.llm
        else:
            raise UnboundLocalError(
                "LLM model is not initialized or is not a type `Pipeline`."
            )

        if self.state.retriever is None:
            raise ValueError("Retriever is not initialized.")

        chain: RunnableSerializable = (
            {
                "context": self.state.retriever | self.__docs_to_jsonstring,
                "question": RunnablePassthrough(),
            }
            | prompt_template
            | {
                "prompt": RunnablePassthrough(),
                "pipeline": lambda _: pipe,
                "config": lambda _: self.config["configurable"],
            }
            | RunnableLambda(self.__run_pipeline)
            | RAFTResponseParser()
        )

        return chain

    def build_invoker(self) -> Iterator[RAFTResponse]:
        questions = self.state.dataset_generator.questions

        if questions is None:
            raise ValueError("Questions are not initialized.")

        chain = self.build_chain()

        def rag_invoke_iterator(questions: IterableDataset):
            pbar = tqdm.tqdm(questions, total=3480)
            for question_dict in pbar:
                pbar.set_description(f"{question_dict['category']}")
                pbar.set_postfix(q=question_dict["question"])
                question = question_dict["question"]
                output = chain.invoke(input=question)

                if not isinstance(output, RAFTResponse):
                    raise ValueError("Output is not a RAFTResponse object.")

                yield output

        return rag_invoke_iterator(questions=questions)
