from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Generic, Optional, TypeVar

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
import orjson

from ..core import State


_YieldType = TypeVar("_YieldType")


class BaseFeature(Generic[_YieldType], metaclass=ABCMeta):
    # Public Properties
    state: State
    config: dict
    runnable: Optional[Runnable] = None

    def __init__(self, state: State, config: dict):
        self.state = state
        self.config = config

    @abstractmethod
    def build_invoker(self) -> Iterator[_YieldType]:
        pass

    @property
    def invoker(self) -> Iterator[_YieldType]:
        return self.build_invoker()

    def __docs_to_jsonstring(self, docs: list[Document]) -> str:
        doc_dicts = list(map(lambda doc: doc.dict(), docs))
        json = orjson.dumps(doc_dicts).decode("utf-8")
        return json
