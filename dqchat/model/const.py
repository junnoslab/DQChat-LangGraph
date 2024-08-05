# HuggingFace
HF_LLM_MODEL_NAME: str = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
HF_EMBEDDING_MODEL_NAME: str = "jhgan/ko-sroberta-multitask"

# Prompt Template
RAG_PROMPT_TEMPLATE: str = """당신은 다이퀘스트의 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.
아래의 context를 활용하여 사용자의 질문에 답을 해. context에 사용자의 질문에 대한 답이 없다면 반드시 "자료가 없어서 답을 드릴 수가 없습니다" 라고 말해.

Context: {context}

"""
