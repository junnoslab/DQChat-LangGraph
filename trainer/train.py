from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import torch
import wandb


_MODEL_ID = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
# _MODEL_ID = "Bllossom/llama-3.1-Korean-Bllossom-405B"
_DATASET_ID = "Junnos/DQChat-raft"

PROMPT_FORMAT: str = """###Question
{question}

###Prompt
{prompt}

{output}
"""


def train():
    wandb.init(
        project="DQChat-Trainer",
        config={
            "learning_rate": 5e-4,
            "batch_size": 8,
            "num_epochs": 800,
            "model_id": _MODEL_ID,
            "dataset_id": _DATASET_ID,
        },
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    tokenizer.padding_side = "right"

    # Load Model
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        # quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load dataset
    dataset = load_dataset(
        path=_DATASET_ID,
        name="question-answer",
        split="train",
    )

    print(dataset.num_rows)

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
    # lora_config = LoraConfig(
    #     task_type="CAUSAL_LM",
    #     r=32,
    #     lora_alpha=64,
    #     lora_dropout=0.1,
    #     bias="lora_only",
    # )

    training_args = SFTConfig(
        output_dir="output",
        num_train_epochs=800,
        per_device_train_batch_size=8,
        max_seq_length=2048,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="paged_adamw_8bit",
        dataloader_num_workers=16,
        save_strategy="steps",
        save_steps=200,
        logging_strategy="steps",
        logging_steps=10,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        formatting_func=format_prompt,
        # peft_config=lora_config,
    )
    trainer.train()

    wandb.finish()
