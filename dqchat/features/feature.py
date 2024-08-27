from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Generic, Optional, TypeVar

from langchain_core.runnables import Runnable

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
        """
        Generates responses from the model in a lazy manner.

        This method should be overridden to define the `invoker`.
        Use the `invoker` property to process the responses obtained from the model.

        The actual processing occurs during iteration, which means this method is 'lazy'.
        This also means that the model's responses are generated on-demand as the returned iterator is consumed.

        The `invoker` can be anything even if it is not a LLM inference.
        For example, it can be a retriever, a pipeline, or just a single method like `concat`.

        :return: `Iterator` which yields individual responses from given callable. The underlying
                  process runs only when this iterator is being iterated over.
        """
        pass

    @property
    def invoker(self) -> Iterator[_YieldType]:
        """
        Iterator for invoking the model and generating responses.

        This property returns an iterator that yields responses from the model.
        It is typically used in conjunction with the `build_invoker` method.

        The returned iterator is 'lazy', meaning that the acutal invocation and response generation occur only when
        the iteration is consumed.This allows for efficient processing of large or potentially infinite streams of data.
        """
        invoker = self.build_invoker()
        return invoker
