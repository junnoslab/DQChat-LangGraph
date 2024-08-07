# HuggingFace
# Dataset
HF_DATASET_NAME: str = "Junnos/DQChat-raft"
HF_CONFIG_QUESTIONS: str = "questions"
HF_QUESTIONS_DATASET_CONTENT_COL: str = "question"
# Model
HF_LLM_MODEL_NAME: str = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
HF_EMBEDDING_MODEL_NAME: str = "jhgan/ko-sroberta-multitask"

# VectorDB
CHROMA_DB_PATH: str = "db"
CHROMA_COLLECTION_NAME: str = "dqchat"

# Prompt Template
RAG_PROMPT_TEMPLATE: str = """당신은 다이퀘스트의 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.
아래의 context를 활용하여 사용자의 질문에 답을 해. context에 사용자의 질문에 대한 답이 없다면 반드시 "자료가 없어서 답을 드릴 수가 없습니다" 라고 말해.

Context: {context}

"""

# Output Parser
POOR_DISTANCE_THRESHOLD: float = 60.0

PROMPT_SYSTEM: str = "System"
PROMPT_CONTEXT: str = "Context"
PROMPT_HUMAN: str = "Human"
PROMPT_ASSISTANT: str = "Assistant"
PROMPT_SECTIONS: list[str] = [PROMPT_SYSTEM, PROMPT_CONTEXT, PROMPT_HUMAN, PROMPT_ASSISTANT]