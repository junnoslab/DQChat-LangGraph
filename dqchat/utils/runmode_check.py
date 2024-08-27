from .runmode import RunMode
from ..core.dataclass.state import State


def check_runmode(state: State, config: dict) -> RunMode:
    return config["configurable"]["run_mode"]
