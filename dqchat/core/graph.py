from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from .nodes import Nodes
from .state import State, Config
from ..validator import validate


class GraphBuilder:
    def __init__(self):
        self.graph = StateGraph(State, config_schema=Config)

    def build(self) -> CompiledStateGraph:
        # Add nodes to the graph
        for node in Nodes:
            self.graph.add_node(*node.node_action_binding)

        # Add edges to the graph
        # Start -> Question Dataset loader
        self.graph.add_edge(start_key=START, end_key=Nodes.QUESTIONS_LOADER.key)
        # Question Dataset loader -> VectorDB context retriever
        self.graph.add_edge(
            start_key=Nodes.QUESTIONS_LOADER.key, end_key=Nodes.RETRIEVER_PREPARER.key
        )
        # VectorDB context retriever -> QA LLM
        self.graph.add_edge(
            start_key=Nodes.RETRIEVER_PREPARER.key, end_key=Nodes.QUESTION_ANSWERER.key
        )
        # QA LLM -> Answer Validator LLM
        self.graph.add_edge(
            start_key=Nodes.QUESTION_ANSWERER.key, end_key=Nodes.ANSWER_VALIDATOR.key
        )
        # Answer Validator LLM -> if invalid: QA LLM, if valid: END
        # https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges
        self.graph.add_conditional_edges(
            source=Nodes.ANSWER_VALIDATOR.key,
            path=validate,
            path_map={"invalid": END, "valid": END},
        )

        # Compile the graph
        compiled_graph = self.graph.compile()
        return compiled_graph
