from PIL import Image
import io

from .core import GraphBuilder


def main():
    graph_builder = GraphBuilder()
    graph = graph_builder.build()

    image_data = graph.get_graph().draw_mermaid_png()
    image = Image.open(fp=io.BytesIO(image_data))
    image.show()


if __name__ == "__main__":
    main()
