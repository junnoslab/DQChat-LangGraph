from argparse import ArgumentParser, Namespace
from PIL import Image
import io

from .core import default_config
from .core.graph import GraphBuilder
from .utils.runmode import RunMode


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", dest="mode", type=str, choices=RunMode.__args__, required=False
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    graph_builder = GraphBuilder()
    graph = graph_builder.build()

    image_data = graph.get_graph().draw_mermaid_png()
    image = Image.open(fp=io.BytesIO(image_data))
    image.save("graph.png")

    graph.invoke(
        input={
            "question_answer": {"question": ""},
            "dataset_generator": {},
        },
        config=default_config(run_mode=args.mode),
    )


if __name__ == "__main__":
    main()
