import argparse

from sparql_kgqa.sparql.utils import SimilarityIndex, load_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "examples",
        type=str,
        nargs="+",
        help="Path to examples file"
    )
    parser.add_argument("out", type=str, help="Output file")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bar"
    )
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> None:
    sim = SimilarityIndex()
    for path in args.examples:
        examples = load_examples(path)
        sim.add(examples, args.batch_size, args.progress)
    sim.save(args.out)


if __name__ == "__main__":
    prepare(parse_args())
