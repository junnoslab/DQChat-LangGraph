from dqchat.core import State


state = State()

state.dataset_generator.next_id = 42

print(state.dataset_generator.next_id)
