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
    WikidataManager,
    WikidataPropertyMapping,
    load_index_and_mapping
)


def get_prompt(kg: str) -> str:
    return f"""\
Generate a SPARQL query with natural language entities and \
properties over {kg}:
"""


def prepare_file(
    file: str,
    files: dict[str, TextIO],
    manager: KgManager,
    seen: set[str],
    sources: list[str],
    args: argparse.Namespace
) -> tuple[int, int, int, int]:
    num_total = 0
    num_duplicate = 0
    num_incomplete = 0
    num_invalid = 0

    with open(file, "r") as f:
        _ = next(f)  # forward headers
        for line in tqdm(
            f,
            desc=f"processing {os.path.basename(file)}",
            disable=not args.progress,
            leave=False
        ):
            sparql, _, source, _ = line.rstrip("\r\n").split("\t")
            if source not in sources:
                continue

            sparql = clean(unquote_plus(sparql))
            num_total += 1
            if sparql in seen:
                num_duplicate += 1
                continue

            seen.add(sparql)

            try:
                sparql_raw = manager.fix_prefixes(sparql)
                sparql_natural, inc = manager.replace_iris(sparql_raw)
            except Exception:
                num_invalid += 1
                continue

            num_incomplete += inc

            files[source].write(sparql + "\n")
            files[f"{source}_input"].write(json.dumps({
                "role": "user",
                "text": get_prompt("wikidata")
            }) + "\n")

            files[f"{source}_natural"].write(
                json.dumps(sparql_natural) + "\n"
            )
            files[f"{source}_raw"].write(sparql_raw + "\n")

    return num_total, num_duplicate, num_incomplete, num_invalid


def prepare(args: argparse.Namespace):
    sources = []
    if not args.robotic_only:
        sources.append("organic")
    if not args.organic_only:
        sources.append("robotic")

    files = {}
    for source in sources:
        if any(
            os.path.exists(os.path.join(args.output_dir, f"{source}{ext}.txt"))
            for ext in ["", ".nl", ".raw"]
        ):
            print(
                f"output files for {source} in {args.output_dir}"
                " already exist"
            )
            return

    ent_dir, ent_type = args.entities
    ent_index, ent_mapping = load_index_and_mapping(ent_dir, ent_type)

    prop_dir, prop_type = args.properties
    prop_index, prop_mapping = load_index_and_mapping(
        prop_dir,
        prop_type,
        WikidataPropertyMapping
    )
    assert isinstance(prop_mapping, WikidataPropertyMapping)

    manager = WikidataManager(
        ent_index,
        prop_index,
        ent_mapping,
        prop_mapping
    )

    for source in sources:
        files[source] = open(
            os.path.join(args.output_dir, f"{source}.txt"), "w"
        )
        files[f"{source}_input"] = open(
            os.path.join(args.output_dir, f"{source}.input.txt"), "w"
        )
        files[f"{source}_natural"] = open(
            os.path.join(args.output_dir, f"{source}.nl.txt"), "w"
        )
        files[f"{source}_raw"] = open(
            os.path.join(args.output_dir, f"{source}.raw.txt"), "w"
        )

    num_total = 0
    num_duplicate = 0
    num_incomplete = 0
    num_invalid = 0
    seen = set()

    for file in tqdm(
        args.files,
        desc="processing files",
        leave=False,
        disable=not args.progress
    ):
        total, duplicate, incomplete, invalid = prepare_file(
            file,
            files,
            manager,
            seen,
            sources,
            args
        )
        num_total += total
        num_duplicate += duplicate
        num_incomplete += incomplete
        num_invalid += invalid

    for f in files.values():
        f.close()

    print(
        f"{num_duplicate:,} / {num_total:,} duplicate "
        f"({num_duplicate / num_total:.1%})"
    )
    print(
        f"{num_incomplete:,} / {num_total:,} incomplete "
        f"({num_incomplete / num_total:.1%})"
    )
    print(
        f"{num_invalid:,} / {num_total:,} invalid "
        f"({num_invalid / num_total:.1%})"
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
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--organic-only", action="store_true")
    source.add_argument("--robotic-only", action="store_true")
    parser.add_argument("--entities", type=str, nargs=2, required=True)
    parser.add_argument("--properties", type=str, nargs=2, required=True)
    parser.add_argument("--rec-limit", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.setrecursionlimit(args.rec_limit)
    prepare(args)
