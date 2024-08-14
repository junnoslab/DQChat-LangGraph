from PIL import Image
import io

from .core.graph import GraphBuilder
from .core.state import Config


def main():
    graph_builder = GraphBuilder()
    graph = graph_builder.build()

    image_data = graph.get_graph().draw_mermaid_png()
    image = Image.open(fp=io.BytesIO(image_data))
    image.save("graph.png")

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
