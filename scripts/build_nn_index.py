import argparse
import os
import pprint
import json
import time

from sparql_kgqa import nn_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "out",
        type=str,
        help="Output index file"
    )
    parser.add_argument(
        "data",
        type=str,
        nargs="+",
        help="Data files to index"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--progress",
        action="store_true"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true"
    )
    parser.add_argument(
        "--example-query",
        type=str,
        default="What is the capital of France?"
    )
    return parser.parse_args()


def build(args: argparse.Namespace) -> None:
    if not os.path.exists(args.out) or args.overwrite:
        index = nn_index.NnIndex(args.device)
        for d in args.data:
            print(f"Adding data from {d}")
            with open(d, "r") as inf:
                samples = []
                for line in inf:
                    sample = json.loads(line)
                    assert "question" in sample and "sparql" in sample
                    samples.append((sample["question"], sample["sparql"]))
                index.add(samples, args.batch_size, args.progress)

        print(
            f"Index info: {len(index.samples):,} samples, "
            f"{index.embeddings.shape} embeddings"
        )
        start = time.perf_counter()
        index.save(args.out)
        end = time.perf_counter()
        print(f"Saved index in {end - start:.2f} seconds")
        del index

    index = nn_index.NnIndex(args.device)
    start = time.perf_counter()
    index.load(args.out)
    end = time.perf_counter()
    print(f"Loaded index in {end - start:.2f} seconds")
    start = time.perf_counter()
    matches = index.find_matches(args.example_query)
    end = time.perf_counter()
    print(f"Found {len(matches)} matches in {1000 * (end - start):.2f} ms:")
    pprint.pprint(matches)


if __name__ == "__main__":
    build(parse_args())
