from typing import Literal

from ..core.state import State


def validate(state: State) -> Literal["valid", "invalid"]:
    """
    Validate the state
    :param state: The state to validate
    :return: "valid" if the state is valid, "invalid" otherwise
    """
    if messages := state.get("messages", []):
        print("Valid")
        return "valid"
    else:
        print("Invalid")
        return "invalid"


def test(state: State) -> State:
    print(state.get("messages", []))
    return state
