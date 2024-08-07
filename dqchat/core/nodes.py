from enum import StrEnum

from langchain_core.runnables.base import RunnableLike

from ..data import load_questions, prepare_retriever, save
from ..model import generate_raft_dataset
from ..validator.validator import test, validate


class Nodes(StrEnum):
    """
    Nodes in the state graph.\n
    Their identifiers are defined with prefix for each node type.\n
    - `ds` for dataset nodes
    - `lm` for language model nodes
    - `vt` for vector database nodes
    """

    QUESTIONS_LOADER = "ds_loader_questions"
    """Questions dataset loader node"""
    RETRIEVER_PREPARER = "vt_preparer_retriever"
    """VectorDB retriever preparation node"""
    QUESTION_ANSWERER = "lm_answerer_question"
    """Question answering language model inference node"""
    ANSWER_VALIDATOR = "lm_validator_answer"
    """Answer validation language model inference node"""

    @property
    def key(self) -> str:
        """
        Get the key of the node.\n
        :return: key of the node
        """
        return self.value

    @property
    def runnable(self) -> RunnableLike:
        """
        Get the runnable for the node.\n
        :return: Runnable for the node
        """
        if self is Nodes.QUESTIONS_LOADER:
            return load_questions
        elif self is Nodes.RETRIEVER_PREPARER:
            return prepare_retriever
        elif self is Nodes.QUESTION_ANSWERER:
            return generate_raft_dataset
        elif self is Nodes.ANSWER_VALIDATOR:
            return save
        else:
            raise ValueError(f"Runnable for node {self} is not defined.")

    @property
    def node_action_binding(self) -> tuple[str, RunnableLike]:
        """
        Get the id and runnable for the node.\n
        :return: Tuple of id and runnable for the node
        """
        return self.key, self.runnable
