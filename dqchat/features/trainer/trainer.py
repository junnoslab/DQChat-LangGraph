from collections.abc import Iterator
import logging

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.pipelines import Pipeline
from trl import SFTConfig, SFTTrainer

from ...core import State
from ...utils.type_helper import guard_let
from ..feature import BaseFeature
from .const import PROMPT_TEMPLATE


_LOGGER = logging.getLogger(__file__)


def prepare_train(state: State, config: dict) -> State:
    trainer = Trainer(state=state, config=config)
    invoker = trainer.invoker

    state.trainer.invoker = invoker
    return state


def train(state: State, config: dict) -> State:
    invoker = state.trainer.invoker

    if invoker is None:
        raise ValueError("No invoker to run.")

    pipe = guard_let(state.llm, Pipeline)
    model = guard_let(pipe.model, PreTrainedModel)
    tokenizer = guard_let(pipe.tokenizer, PreTrainedTokenizerBase)
    dataset = guard_let(state.trainer.dataset, Dataset)

    def format_prompt(input_dataset: Dataset) -> list[str]:
        prompts: list[str] = []

        for i in range(len(input_dataset)):
            question = input_dataset["question"][i]
            context = input_dataset["context"][i]
            reason = input_dataset["reason"][i]
            answer = input_dataset["answer"][i]

            formatted_prompt = PROMPT_TEMPLATE.format(
                question=question,
                context=context,
                reason=reason,
                answer=answer,
            )
            prompts.append(formatted_prompt)

        return prompts

    training_args = SFTConfig(
        output_dir="output/train",
        num_train_epochs=800,
        per_device_train_batch_size=4,
        max_seq_length=2048,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        # optim="paged_adamw_8bit",
        dataloader_num_workers=8,
        save_strategy="steps",
        save_steps=500,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        formatting_func=format_prompt,
    )
    trainer.train()

    return state


class Trainer(BaseFeature[int]):
    def build_invoker(self) -> Iterator[int]:
        iterator = iter(range(1_000))
        return iterator
