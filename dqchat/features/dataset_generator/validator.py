from typing import Literal

from ...core import State


def validate(state: State, config: dict) -> Literal["pass", "fail"]:
    """
    Validates the dataset generated by the feature `DatasetBuilder`.

    TODO: This function is not implemented yet. Returning "pass" for now.
    """
    return "pass"
