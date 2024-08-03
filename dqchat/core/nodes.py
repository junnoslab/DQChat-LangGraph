from enum import StrEnum

from langchain_core.runnables.base import RunnableLike


def run():
    print("skldfj")


class Nodes(StrEnum):
    """
    Nodes in the state graph.\n
    Their identifiers are defined with prefix for each node type.\n
    - `ds` for dataset nodes
    - `lm` for language model nodes
    """
    QUESTIONS_LOADER = "ds_loader_questions"
    """Questions dataset loader node"""
    QUESTION_ANSWERER = "lm_answerer_question"
    """Question answering language model inference node"""
    ANSWER_PARSER = "ds_parser_answer"
    """Pydantic model to dataset parser node"""
    ANSWER_VALIDATOR = "lm_validator_answer"
    """Answer validation language model inference node"""

    @property
    def id(self) -> str:
        """
        Get the ID of the node.\n
        :return: ID of the node
        """
        return self.value

    @property
    def runnable(self) -> RunnableLike:
        """
        Get the runnable for the node.\n
        :return: Runnable for the node
        """
        if self is Nodes.QUESTIONS_LOADER:
            return run
        elif self is Nodes.QUESTION_ANSWERER:
            return run
        elif self is Nodes.ANSWER_PARSER:
            return run
        elif self is Nodes.ANSWER_VALIDATOR:
            return run
        else:
            raise ValueError(f"Runnable for node {self} is not defined.")

    @property
    def node_action_binding(self) -> tuple[str, RunnableLike]:
        """
        Get the id and runnable for the node.\n
        :return: Tuple of id and runnable for the node
        """
        return self.id, self.runnable
