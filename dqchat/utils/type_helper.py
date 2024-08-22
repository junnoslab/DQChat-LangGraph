from typing import TypeVar, Type, Any


__T = TypeVar("__T")


def guard_type(value: Any, expected_type: Type[__T]) -> __T:
    if isinstance(value, expected_type):
        return value
    else:
        actual_type = type(value).__name__
        raise TypeError(
            f"Type mismatch: Expected {expected_type.__name__}, but received {actual_type}."
            f"Value: {repr(value)}"
        )
