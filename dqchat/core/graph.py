from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from .nodes import Nodes
from .state import State


class GraphBuilder:
    def __init__(self):
        self.graph = StateGraph(State)

    def build(self) -> CompiledStateGraph:
        for node in Nodes:
            self.graph.add_node(*node.node_action_binding)
        self.graph.add_edge(start_key=START, end_key=Nodes.QUESTIONS_LOADER.id)
        self.graph.add_edge(start_key=Nodes.QUESTIONS_LOADER.id, end_key=Nodes.QUESTION_ANSWERER.id)
        self.graph.add_edge(start_key=Nodes.QUESTION_ANSWERER.id, end_key=Nodes.ANSWER_PARSER.id)
        self.graph.add_edge(start_key=Nodes.ANSWER_PARSER.id, end_key=Nodes.ANSWER_VALIDATOR.id)
        self.graph.add_edge(start_key=Nodes.ANSWER_VALIDATOR.id, end_key=END)

        compiled_graph = self.graph.compile()
        return compiled_graph
