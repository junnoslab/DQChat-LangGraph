from langgraph.graph import START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from . import Config, State
from .nodes import Nodes


class GraphBuilder:
    def __init__(self):
        self.graph = StateGraph(state_schema=State, config_schema=Config)

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
                "raft_dataset": Nodes.RF_MODEL_LOADER.key,
                "inference": Nodes.INFERENCE_PREPARER.key,
                "train": Nodes.TR_MODEL_LOADER.key,
            },
        )
        # Dataset generator
        self.graph.add_edge(
            start_key=Nodes.RF_MODEL_LOADER.key,
            end_key=Nodes.RF_QUESTIONS_LOADER.key,
        )
        self.graph.add_edge(
            start_key=Nodes.RF_QUESTIONS_LOADER.key,
            end_key=Nodes.RF_INVOKER_CHAIN_BUILDER.key,
        )
        self.graph.add_edge(
            start_key=Nodes.RF_INVOKER_CHAIN_BUILDER.key,
            end_key=Nodes.RF_QA_INVOKER.key,
        )
        self.graph.add_conditional_edges(
            source=Nodes.RF_QA_INVOKER.key,
            path=Nodes.RF_ANSWER_VALIDATOR.runnable,
            path_map={
                "pass": Nodes.RF_QA_DATASET_CHECKPOINTER.key,
                "fail": Nodes.RF_QUESTIONS_LOADER.key,
            },
        )
        self.graph.add_edge(start_key=Nodes.RF_QA_DATASET_CHECKPOINTER.key, end_key=END)
        # Inference
        self.graph.add_edge(
            start_key=Nodes.INFERENCE_PREPARER.key, end_key=Nodes.INPUT_RETRIEVER.key
        )
        self.graph.add_conditional_edges(
            source=Nodes.INPUT_RETRIEVER.key,
            path=Nodes.INPUT_VALIDATOR.runnable,
            path_map={
                "next": Nodes.RESULT_INFERENCER.key,
                "exit": END,
            },
        )
        self.graph.add_edge(
            start_key=Nodes.RESULT_INFERENCER.key, end_key=Nodes.INFERENCE_PREPARER.key
        )

        # Compile the graph
        compiled_graph = self.graph.compile()
        return compiled_graph
