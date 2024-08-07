from PIL import Image
import io

from .core.graph import GraphBuilder
from .core.state import Config


LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="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"
LANGCHAIN_PROJECT="DQChat"


def main():
    graph_builder = GraphBuilder()
    graph = graph_builder.build()

    # image_data = graph.get_graph().draw_mermaid_png()
    # image = Image.open(fp=io.BytesIO(image_data))
    # image.show()

    graph.invoke(
        {
            "questions": None,
            "retriever": None,
            "responses": [],
        },
        config=Config.default_config(),
    )


if __name__ == "__main__":
    main()
