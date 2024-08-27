from typing import Literal

from langchain_core.documents import Document
import orjson

from ..const import RAG_PROMPT_TEMPLATE, USER_TEMPLATE
from ..core.dataclass.state import State


def prepare_for_inference(state: State, config: dict) -> State:
    if state.llm is not None and state.sampling_params is not None:
        return state

    # Prepare LLM for inference
    llm = LLM(
        model="Junnos/DQChat",
        # dtype=torch.bfloat16,
        # enforce_eager=True,
        # quantization="bitsandbytes",
        # load_format="bitsandbytes",
        gpu_memory_utilization=0.6,
    )

    # Prepare tokenizer
    tokenizer = llm.get_tokenizer()
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    # Prepare sampling parameters
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=4096,
        stop_token_ids=terminators,
    )

    state.llm = llm
    state.sampling_params = sampling_params

    return state


def retrieve_input(state: State, config: dict) -> State:
    question = input("질문을 입력하세요: ")
    state.question_answer.question = question
    return state


def validate_input(state: State, config: dict) -> Literal["exit", "next"]:
    question = state.question_answer.question
    if question == "exit" or question == "":
        return "exit"
    return "next"


def inference(state: State, config: dict) -> State:
    qa_state = state.question_answer
    tokenizer = state.llm.get_tokenizer()

    context = __format_docs(state.retriever.invoke(qa_state.question))
    json = orjson.loads(context)
    answers = [item["metadata"]["answer"] for item in json]
    context = ", ".join(answers)

    messages = [
        {"role": "system", "content": RAG_PROMPT_TEMPLATE.format(context=context)},
        {"role": "user", "content": USER_TEMPLATE.format(question=qa_state.question)},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    outputs = state.llm.generate(prompts=prompt, sampling_params=state.sampling_params)

    for output in outputs:
        output = output.outputs[0]
        print(output)

    return state


def __format_docs(docs: list[Document]) -> str:
    _doc_dicts = list(map(lambda doc: doc.dict(), docs))
    _json = orjson.dumps(_doc_dicts).decode("utf-8")
    return _json
