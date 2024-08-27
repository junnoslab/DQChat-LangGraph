from argparse import ArgumentParser, Namespace
from PIL import Image
import io
import logging

from .core import default_config
from .core.graph import GraphBuilder
from .utils.runmode import RunMode


def setup_logger():
    logging.basicConfig(level=logging.INFO)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", dest="mode", type=str, choices=RunMode.__args__, required=False
    )
    parser.add_argument(
        "--dataset-config", dest="dataset_config", type=str, required=False
    )
    parser.add_argument("--cache-dir", dest="cache_dir", type=str, required=False)

    args = parser.parse_args()

    return args


def main():
    setup_logger()

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
            "trainer": {},
        },
        config=default_config(
            run_mode=args.mode,
            dataset_config=args.dataset_config,
            model_cache_path=args.cache_dir,
        ),
    )


if __name__ == "__main__":
    main()
