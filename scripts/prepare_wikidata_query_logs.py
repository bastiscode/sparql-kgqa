import argparse
import json
import sys
import os
from typing import TextIO
from urllib.parse import unquote_plus

from tqdm import tqdm

from text_utils import grammar

from sparql_kgqa.sparql.utils import (
    KgIndex,
    general_prefixes,
    load_kg_index,
    clean,
    fix_prefixes,
    load_sparql_parser,
    replace_vars_and_special_tokens,
    replace_entities_and_properties
)


def get_prompt(kg: str) -> str:
    return json.dumps(f"""\
Task:
SPARQL query autocompletion over the specified knowledge graphs

Knowledge graphs:
{kg}

SPARQL:
""")


def prepare_file(
    file: str,
    files: dict[str, TextIO],
    ent_index: KgIndex,
    prop_index: KgIndex,
    parser: grammar.LR1Parser,
    seen: set[str],
    sources: list[str],
    args: argparse.Namespace
) -> tuple[int, int, int, int]:
    num_total = 0
    num_duplicate = 0
    num_incomplete = 0
    num_invalid = 0

    prefixes = general_prefixes()
    prefixes.update(ent_index.prefixes)
    prefixes.update(prop_index.prefixes)

    ent_indices = {"wikidata": ent_index}
    prop_indices = {"wikidata": prop_index}

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
                sparql_raw = fix_prefixes(
                    sparql,
                    parser,
                    prefixes
                )
            except Exception:
                num_invalid += 1
                continue

            sparql_raw = replace_vars_and_special_tokens(
                sparql_raw,
                parser,
                args.version
            )

            sparqls_natural, inc = replace_entities_and_properties(
                sparql,
                parser,
                ent_indices,
                prop_indices,
                args.version,
                "in_order"
            )
            num_incomplete += inc

            for sparql_natural in sparqls_natural:
                files[source].write(sparql + "\n")
                files[f"{source}_input"].write(get_prompt("wikidata") + "\n")

                sparql_natural = fix_prefixes(
                    sparql_natural,
                    parser,
                    prefixes
                )
                sparql_natural = replace_vars_and_special_tokens(
                    sparql_natural,
                    parser,
                    args.version
                )

                files[f"{source}_natural"].write(sparql_natural + "\n")
                files[f"{source}_raw"].write(sparql_raw + "\n")

    return num_total, num_duplicate, num_incomplete, num_invalid


def prepare(args: argparse.Namespace):
    parser = load_sparql_parser(["wikidata"])

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

    entity_index = load_kg_index(
        args.entity_index,
        args.entity_redirects,
        args.entity_prefixes,
        args.progress
    )

    property_index = load_kg_index(
        args.property_index,
        prefixes_path=args.property_prefixes,
        progress=args.progress
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
            entity_index,
            property_index,
            parser,
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
    parser.add_argument("--entity-index", type=str, required=True)
    parser.add_argument("--entity-redirects", type=str, default=None)
    parser.add_argument("--entity-prefixes", type=str, default=None)
    parser.add_argument("--property-index", type=str, required=True)
    parser.add_argument("--property-prefixes", type=str, default=None)
    parser.add_argument("--version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--rec-limit", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.setrecursionlimit(args.rec_limit)
    prepare(args)
