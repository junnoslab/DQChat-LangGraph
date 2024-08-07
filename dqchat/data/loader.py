from datasets import IterableDataset, load_dataset

from ..core.state import State


def load_questions(state: State, config: dict) -> State:
    """
    Load questions from the dataset.\n
    :return: State with questions loaded
    """
    config = config["configurable"]
    dataset: IterableDataset = load_dataset(
        path=config["dataset_name"],
        name=config["dataset_config"],
        split="train",
        token=config["hf_access_token"],
        streaming=True,
    )

    state["questions"] = dataset
    return state
