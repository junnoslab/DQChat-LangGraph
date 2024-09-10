from typing import TypeVar, Type, Any


__T = TypeVar("__T")


def guard_let(value: Any, expected_type: Type[__T]) -> __T:
    if isinstance(value, expected_type):
        return value
    elif value is None:
        raise TypeError(
            f"Type mismatch: Expected solid type '{expected_type.__name__}', but received 'None'.\n"
        )
    else:
        actual_type = type(value).__name__
        raise TypeError(
            f"Type mismatch: Expected '{expected_type.__name__}', but received {actual_type}.\n"
            f"Value: {repr(value)}"
        )


# TODO: Add `None` type unwrapper.
