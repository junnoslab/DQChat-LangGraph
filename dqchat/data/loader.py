from datasets import IterableDataset, load_dataset

from ..core import State


def load_questions(state: State, config: dict) -> State:
    """
    Load questions from the dataset.\n
    :return: State with questions loaded
    """
    config = config["configurable"]
    dataset = load_dataset(
        path=config["dataset_name"],
        name=config["dataset_config"],
        cache_dir=config["dataset_cache_path"],
        split="train",
        token=config["hf_access_token"],
        streaming=True,
    )

    if not isinstance(dataset, IterableDataset):
        raise ValueError("Dataset is not an IterableDataset.")

    state.dataset_generator.questions = dataset
    return state
