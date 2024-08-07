from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import torch


_MODEL_ID = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
_DATASET_ID = "Junnos/DQChat-raft"


def train():
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        _MODEL_ID,
        device_map="auto",
    )

    # Load Model
    bnb_config = BitsAndBytesConfig(
		load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
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
    def format_prompt(dataset: Dataset):
        return {
			"input_ids": dataset["question"],
			"labels": dataset["answer"],
		}

    # Setup data collator
    data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template="Complete the following text:",
        response_template="The next word is: ",
        tokenizer=tokenizer,
	)

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
        train_dataset=dataset,
        formatting_func=format_prompt,
        data_collator=data_collator,
        peft_config=lora_config,
	)
    trainer.train()
