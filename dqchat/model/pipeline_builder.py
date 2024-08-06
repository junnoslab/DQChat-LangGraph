from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer
import orjson
import torch

from ..const import RAG_PROMPT_TEMPLATE
from ..core.state import State


def __build_pipeline(model_id: str) -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    pipeline = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        device=-1,
        model_kwargs=dict(torch_dtype=torch.bfloat16),
        pipeline_kwargs=dict(
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            eos_token_id=terminators,
        ),
    )

    return pipeline


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


def generate_raft_dataset(state: State, config: dict) -> State:
    # Prepare pipeline
    pipeline = __build_pipeline(config["configurable"]["model_name"])

    # Prepare prompt template
    prompt_template = __prepare_prompt_template()

    # Prepare retriever
    retriever = state["retriever"]
    if retriever is None:
        raise ValueError("Retriever is not set.")

    # Chain pipeline + retriever + prompt
    rag_chain = (
        {"context": retriever | __format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | pipeline
        | StrOutputParser()
    )

    # Prepare questions dataset
    for question in state["questions"]:
        output = rag_chain.invoke(input=question.page_content)
        print(output)

    return state


def __format_docs(docs: list[Document]) -> str:
    _doc_dicts = list(map(lambda doc: doc.dict(), docs))
    _json = orjson.dumps(_doc_dicts).decode("utf-8")
    return _json
