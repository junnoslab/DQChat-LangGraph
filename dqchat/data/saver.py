from datasets import Dataset
import pandas as pd

from ..core.state import State


def save(state: State, config: dict) -> State:
    responses = state["responses"]

    data_dicts = [response.model_dump() for response in responses]
    df = pd.DataFrame(data_dicts)
    dataset = Dataset.from_pandas(df)

    dataset.to_csv("responses.csv")
    
    return state
