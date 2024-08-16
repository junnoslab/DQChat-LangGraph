from typing import Literal

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.documents import Document
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import orjson
import torch

from ..const import RAG_PROMPT_TEMPLATE
from ..core.state import State


def __prepare_prompt_template() -> ChatPromptTemplate:
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


def retrieve_input(state: State, config: dict) -> Literal["exit", "next"]:
    question = input("질문을 입력하세요: ")

    if question == "exit":
        state["question_for_inference"] = None
        return "exit"

    state["question_for_inference"] = question
    return "next"


def prepare_for_inference(state: State, config: dict) -> State:
    if (
        state["llm_for_inference"] is not None
        and state["sampling_params_for_inference"] is not None
    ):
        return state

    config = config["configurable"]
    llm_model_name: str = config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    llm = LLM(
        model="output/t1/checkpoint-1400",
        dtype=torch.bfloat16,
        enforce_eager=True,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        gpu_memory_utilization=0.7,
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=2048,
        stop_token_ids=terminators,
    )

    state["llm_for_inference"] = llm
    state["sampling_params_for_inference"] = sampling_params

    return state


def inference(state: State, config: dict) -> State:
    retriever = state["retriever"]
    llm = state["llm_for_inference"]
    sampling_params = state["sampling_params_for_inference"]

    prompt = state.get("question_for_inference", "")
    context = __format_docs(retriever.invoke(prompt))
    prompt_template = __prepare_prompt_template()
    prompt_template.bind(context=context, question=prompt)
    outputs = llm.generate(prompts=prompt, sampling_params=sampling_params)

    for output in outputs:
        print(output.outputs[0].text)

    return state


def __format_docs(docs: list[Document]) -> str:
    _doc_dicts = list(map(lambda doc: doc.dict(), docs))
    _json = orjson.dumps(_doc_dicts).decode("utf-8")
    return _json
