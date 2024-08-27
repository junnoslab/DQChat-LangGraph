from ..core import State


def save(state: State, config: dict) -> State:
    dataset = state.dataset_generator.responses

    if dataset is None:
        raise ValueError("No responses to save.")

    # data_dicts = [response.model_dump() for response in responses]
    # df = pd.DataFrame(data_dicts)
    # dataset = Dataset.from_pandas(df)

    dataset.to_csv("responses.csv")

    return state
