from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.documents import Document
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from transformers import AutoTokenizer
from transformers.pipelines import Pipeline, pipeline
import orjson
import torch
import tqdm

from ..const import RAG_PROMPT_TEMPLATE
from ..core.state import State
from ..data.parser import RAFTResponseParser


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


def __generate_pipeline(model_name: str) -> Pipeline:
    pipe = pipeline(
        task="text-generation",
        model=model_name,
        device_map="auto",
        model_kwargs={
            "torch_dtype": torch.bfloat16,
        },
    )

    return pipe


def __run_pipeline(inputs: dict) -> str:
    pipeline: Pipeline = inputs["pipeline"]
    prompt: ChatPromptValue = inputs["prompt"]
    config = inputs["config"]

    llm_model_name: str = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    output = pipeline(
        prompt.to_string(),
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        eos_token_id=terminators,
    )

    return output[0]["generated_text"]


def generate_raft_dataset(state: State, config: dict) -> State:
    # Unwrap state and config
    config = config["configurable"]
    llm_model_name: str = config["model_name"]

    # Prepare pipeline
    pipeline: Pipeline = __generate_pipeline(model_name=llm_model_name)

    # Prepare prompt template
    prompt_template: ChatPromptTemplate = __prepare_prompt_template()

    # Prepare retriever
    retriever = state["retriever"]

    # Chain pipeline + retriever + prompt
    rag_chain = (
        {"context": retriever | __format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | {
            "prompt": RunnablePassthrough(),
            "pipeline": lambda _: pipeline,
            "config": lambda _: config,
        }
        | RunnableLambda(__run_pipeline)
        | RAFTResponseParser()
    )

    # Prepare questions dataset
    pbar = tqdm.tqdm(state["questions"], total=3480)
    for idx, question_dict in enumerate(pbar):
        pbar.set_description(f"{question_dict['category']}")
        pbar.set_postfix(q=question_dict["question"])
        question = question_dict["question"]
        output = rag_chain.invoke(input=question)

        state["responses"].append(output)

    return state


def __format_docs(docs: list[Document]) -> str:
    _doc_dicts = list(map(lambda doc: doc.dict(), docs))
    _json = orjson.dumps(_doc_dicts).decode("utf-8")
    return _json
