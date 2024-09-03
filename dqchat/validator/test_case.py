from dataclasses import dataclass
from typing import Optional

from ..utils.type_helper import guard_type


@dataclass
class LLMTestCase:
    input: str
    actual_output: str
    expected_output: Optional[str] = None
    contexts: Optional[list[str]] = None

    def __post_init__(self):
        try:
            self.contexts = guard_type(self.contexts, list[str])
        except TypeError as e:
            raise TypeError("'contexts' must be None or a list of strings") from e
