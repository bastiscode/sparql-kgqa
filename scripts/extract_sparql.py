import argparse
import os
import json
import requests

from sparql_kgqa.sparql.utils import QLEVER_URLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        choices=["qlever"],
    )
    parser.add_argument(
        "out",
        type=str,
    )
    return parser.parse_args()


def extract(args: argparse.Namespace):
    kg_samples = {}
    if args.source == "qlever":
        for kg in QLEVER_URLS:
            examples = requests.get(
                f"https://qlever.cs.uni-freiburg.de/api/examples/{kg}"
            ).text
            samples = []
            for line in examples.splitlines():
                query, sparql = line.split("\t")
                samples.append((query, sparql))
            kg_samples[kg] = samples
    else:
        raise ValueError(f"unknown source: {args.source}")

    for kg, samples in kg_samples.items():
        with open(os.path.join(args.out, f"{kg}_input.txt"), "w") as inf, \
                open(os.path.join(args.out, f"{kg}_sparql.txt"), "w") as of, \
                open(os.path.join(args.out, f"{kg}_examples.tsv"), "w") as exf:
            for query, sparql in samples:
                inf.write(
                    json.dumps([{"role": "user", "text": query}]) + "\n"
                )
                of.write(sparql + "\n")
                exf.write(
                    json.dumps({"query": query, "sparql": sparql}) + "\n"
                )


if __name__ == "__main__":
    extract(parse_args())
