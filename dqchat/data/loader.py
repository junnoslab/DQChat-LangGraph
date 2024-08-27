from datasets import Dataset, load_dataset

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
        cache_dir=config.get("dataset_cache_path", None),
        split="train",
        token=config["hf_access_token"],
    )

    if not isinstance(dataset, Dataset):
        raise ValueError("dataset is not a general Dataset.")

    state.dataset_generator.questions = dataset
    return state
