import argparse
import json
import sys
import os
from typing import TextIO
from urllib.parse import unquote_plus

from tqdm import tqdm

from sparql_kgqa.sparql.utils import clean
from sparql_kgqa.sparql.utils2 import (
    KgManager,
    WikidataPropertyMapping,
    get_kg_manager,
    load_index_and_mapping
)


def get_prompt(kg: str) -> str:
    return f"""\
Generate a natural language SPARQL query over {kg}:
"""


def prepare_file(
    in_file: str,
    out_files: dict[str, TextIO],
    seen: set[str],
    args: argparse.Namespace
) -> tuple[int, int]:
    num_total = 0
    num_duplicate = 0

    with open(in_file, "r") as f:
        _ = next(f)  # forward headers
        for line in tqdm(
            f,
            desc=f"processing {os.path.basename(in_file)}",
            disable=not args.progress,
            leave=False
        ):
            sparql, _, source, _ = line.rstrip("\r\n").split("\t")
            if source not in out_files:
                continue

            sparql = clean(unquote_plus(sparql))
            num_total += 1
            if sparql in seen:
                num_duplicate += 1
                continue

            seen.add(sparql)
            out_files[source].write(json.dumps(sparql) + "\n")

    return num_total, num_duplicate


def prepare(args: argparse.Namespace):
    sources = []
    if not args.robotic_only:
        sources.append("organic")
    if not args.organic_only:
        sources.append("robotic")

    files = {}
    for source in sources:
        out_file = os.path.join(args.output_dir, f"{source}_raw.jsonl")
        if os.path.exists(out_file) and not args.overwrite:
            print(
                f"output file for {source} in {args.output_dir}"
                " already exist"
            )
            continue
        files[source] = open(out_file, "w")

    num_total = 0
    num_duplicate = 0
    seen = set()

    for file in tqdm(
        args.files,
        desc="processing files",
        leave=False,
        disable=not args.progress
    ):
        total, duplicate = prepare_file(
            file,
            files,
            seen,
            args
        )
        num_total += total
        num_duplicate += duplicate

    for f in files.values():
        f.close()

    print(
        f"{num_duplicate:,} / {num_total:,} duplicate "
        f"({num_duplicate / num_total:.1%})"
    )
    print(f"sources: {sources}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        type=str,
        required=True
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--progress",
        action="store_true"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--organic-only", action="store_true")
    source.add_argument("--robotic-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args)
