from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from .nodes import Nodes
from .state import State, Config


class GraphBuilder:
    def __init__(self):
        self.graph = StateGraph(State, config_schema=Config)

    def build(self) -> CompiledStateGraph:
        # Add nodes to the graph
        non_conditional_nodes = [node for node in Nodes if not node.is_conditional]
        for node in non_conditional_nodes:
            self.graph.add_node(*node.node_action_binding)

        # Add edges to the graph
        self.graph.add_edge(start_key=START, end_key=Nodes.RETRIEVER_PREPARER.key)
        # https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges
        self.graph.add_conditional_edges(
            source=Nodes.RETRIEVER_PREPARER.key,
            path=Nodes.RUNMODE_CHECKER.runnable,
            path_map={
                "raft_dataset": Nodes.QUESTIONS_LOADER.key,
                "inference": Nodes.RESULT_INFERENCER.key,
            },
        )
        self.graph.add_edge(
            start_key=Nodes.QUESTIONS_LOADER.key, end_key=Nodes.QUESTION_ANSWERER.key
        )
        self.graph.add_edge(
            start_key=Nodes.QUESTION_ANSWERER.key,
            end_key=Nodes.QA_DATASET_CHECKPOINTER.key,
        )
        self.graph.add_edge(start_key=Nodes.QA_DATASET_CHECKPOINTER.key, end_key=END)
        self.graph.add_edge(start_key=Nodes.RESULT_INFERENCER.key, end_key=END)

        # Compile the graph
        compiled_graph = self.graph.compile()
        return compiled_graph
