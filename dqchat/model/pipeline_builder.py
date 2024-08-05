from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.pipelines.pt_utils import KeyDataset
import torch

from .const import RAG_PROMPT_TEMPLATE
from ..core.state import State


def __build_pipeline(model_id: str) -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    pipeline = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        device_map="auto",
        model_kwargs=dict(
            torch_dtype=torch.bfloat16
        ),
        pipeline_kwargs=dict(
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            eos_token_ids=terminators
        )
    )

    return pipeline


def __prepare_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["context"],
                    template=RAG_PROMPT_TEMPLATE
                )
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["question"],
                    template="{question}<|eot_id|>\n\nAssistant:"
                )
            )
        ],
        input_variables=["context", "question"],
    )


def generate_raft_dataset(state: State, config) -> State:
    model_id = config["configurable"]["model_name"]
    pipeline = __build_pipeline(model_id)

    questions = state["questions"]

    # questions_dataset = state["question_dataset"]
    #
    # prompt_template = __prepare_prompt_template()
    #
    # for out in pipeline(KeyDataset(questions_dataset)):
    #     pass
    return state
