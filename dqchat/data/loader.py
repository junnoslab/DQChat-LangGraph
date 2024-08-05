from langchain_community.document_loaders import HuggingFaceDatasetLoader

from ..core.state import State


def load_questions(state: State, config) -> State:
    """
    Load questions from the dataset.\n
    :return: State with questions loaded
    """
    loader = HuggingFaceDatasetLoader(
        path=config["configurable"]["dataset_name"],
        name=config["configurable"]["dataset_config"],
        use_auth_token=config["configurable"]["hf_access_token"],
    )

    state["questions"] = loader.lazy_load()
    return state
