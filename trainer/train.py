from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
import torch


_MODEL_ID = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
_DATASET_ID = "Junnos/DQChat-raft"

PROMPT_FORMAT: str = """###Question
{question}

###Prompt
{prompt}

{output}
"""


def train():
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    tokenizer.padding_side = "right"

    # Load Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load dataset
    dataset = load_dataset(
        path=_DATASET_ID,
        name="question-answer",
        split="train",
    )

    # Configure prompt formatting
    def format_prompt(input_dataset: Dataset) -> list[str]:
        prompts: list[str] = []

        for i in range(len(input_dataset)):
            prompt = input_dataset["prompt"][i]
            question = input_dataset["question"][i]
            output = input_dataset["output"][i]

            formatted_prompt = PROMPT_FORMAT.format(
                question=question,
                prompt=prompt,
                output=output,
            )
            prompts.append(formatted_prompt)

        return prompts

    # Setup LORA configuration
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="lora_only",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=format_prompt,
        peft_config=lora_config,
    )
    trainer.train()
