from typing import Any
import inspect
import orjson
import re

from langchain.pydantic_v1 import BaseModel
from langchain.schema import BaseOutputParser

from ..const import (
    POOR_DISTANCE_THRESHOLD,
    PROMPT_HUMAN,
    PROMPT_SYSTEM,
    PROMPT_CONTEXT,
    PROMPT_ASSISTANT,
    PROMPT_SECTIONS,
)


class RAFTResponse(BaseModel):
    question: str
    """사용자 입력 질문"""
    prompt: str
    """사용자 입력과 VectorDB 검색 결과"""
    doc_ids: list[str]
    """VectorDB 검색 결과"""
    positive_doc_ids: list[str]
    """VectorDB 검색 결과 중 positive"""
    negative_doc_ids: list[str]
    """VectorDB 검색 결과 중 negative"""
    reasons: list[str]
    """출처(Document)"""
    answer: str
    """생성된 답변"""

    @property
    def output(self) -> str:
        """Reason과 Answer의 합본"""
        reasons = f"###Reasons:\n{self.reasons}" if self.reasons != [] else ""
        answer = f"###Answer:\n{self.answer}"
        return "\n\n".join([reasons, answer])

    def model_dump(self, **kwargs) -> dict[str, Any]:
        attribs = super().model_dump(**kwargs)

        # Add all properties to the dictionary
        for name, value in inspect.getmembers(self.__class__):
            if isinstance(value, property) and name[:2] != "__":
                attribs[name] = getattr(self, name)

        # Convert any sets to lists
        for key, value in attribs.items():
            if isinstance(value, set):
                attribs[key] = list(value)

        return attribs


class RAFTResponseParser(BaseOutputParser):
    @staticmethod
    def __parse_contexts(contexts: list[dict[str, Any]]) -> dict[str, Any]:
        doc_ids = [context.get("id", "") for context in contexts]
        positive_doc_ids = [
            context.get("id", "")
            for context in contexts
            if context.get("metadata", {}).get("distance", 0) < POOR_DISTANCE_THRESHOLD
        ]
        negative_doc_ids = [id for id in doc_ids if id not in positive_doc_ids]

        def metadata(context: dict[str, Any], target: str) -> str:
            return context.get("metadata", {}).get(target, "")

        reasons = [
            f"'{metadata(context, 'answer')}'은/는 ({metadata(context, 'category')} - {metadata(context, 'clause')} - {doc_id}) 문서에서 확인할 수 있습니다."
            for context, doc_id in zip(contexts, doc_ids)
            if doc_id in positive_doc_ids
        ]

        return {
            "doc_ids": doc_ids,
            "positive_doc_ids": positive_doc_ids,
            "negative_doc_ids": negative_doc_ids,
            "reasons": reasons,
        }

    @staticmethod
    def __split_sections(text: str) -> list[str]:
        return re.split(r"\n{2,}", text)

    @staticmethod
    def __parse_section_content(section: str, pattern: str) -> dict[str, str]:
        if match := re.match(pattern=pattern, string=section):
            prefix = match.group(1)
            section = re.sub(pattern, "", match.string, flags=re.MULTILINE)
            return {prefix: section}
        return {}

    def parse(self, text: str) -> RAFTResponse:
        sections = RAFTResponseParser.__split_sections(text)

        result: dict[str, Any] = {}

        section_content_pattern = rf"^({'|'.join(PROMPT_SECTIONS)}):\s"
        for section in sections:
            result.update(
                RAFTResponseParser.__parse_section_content(
                    section=section, pattern=section_content_pattern
                )
            )

        # Remove eos_token from Human(prompt)
        result[PROMPT_HUMAN] = re.sub(r"<\|eot_id\|>", "", result[PROMPT_HUMAN])

        try:
            contexts: list[dict[str, Any]] = orjson.loads(
                result.get(PROMPT_CONTEXT, "")
            )
        except orjson.JSONDecodeError:
            print("Failed to parse Contexts: ", result.get(PROMPT_CONTEXT, ""))
            contexts = []

        context_contents = RAFTResponseParser.__parse_contexts(contexts)

        parsed_result = {
            "question": result.get(PROMPT_HUMAN, ""),
            "prompt": "\n".join(
                [result.get(PROMPT_SYSTEM, ""), result.get(PROMPT_CONTEXT, "")]
            ),
            "answer": result[PROMPT_ASSISTANT],
            **context_contents,
        }

        response = RAFTResponse.model_validate(parsed_result)

        return response
