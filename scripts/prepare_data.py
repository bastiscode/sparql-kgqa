import argparse
import os
import re
import json
import collections

from tqdm import tqdm
from datasets import load_dataset

from sparql_kgqa.sparql.utils import (
    KgIndex,
    clean,
    general_prefixes,
    load_sparql_parser,
    fix_prefixes,
    replace_vars_and_special_tokens,
    preprocess_natural_language_query,
    replace_iris
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    data = parser.add_mutually_exclusive_group(required=True)

    # wikidata
    data.add_argument("--wikidata-simple-questions", type=str)
    data.add_argument("--qald-10", type=str)
    data.add_argument("--time-questions", type=str)
    data.add_argument("--cron-questions", type=str)
    data.add_argument("--mkqa", type=str)
    data.add_argument("--mintaka", type=str)
    data.add_argument("--lc-quad2-wikidata", type=str)
    data.add_argument("--mcwq", type=str)
    data.add_argument("--qa-wiki", type=str)
    data.add_argument("--kqa-pro", type=str)

    # freebase
    data.add_argument("--graph-questions", type=str)
    data.add_argument("--wqsp", type=str)
    data.add_argument("--complex-web-questions", type=str)
    data.add_argument("--freebase-simple-questions", type=str)
    data.add_argument("--30mqa", type=str)
    data.add_argument("--cfq", type=str)
    data.add_argument("--grail-qa", type=str)
    data.add_argument("--freebase-qa", type=str)

    # dbpedia
    data.add_argument("--lc-quad2-dbpedia", type=str)
    data.add_argument("--qald-9-plus", type=str)
    data.add_argument("--simple-dbpedia-qa", type=str)
    data.add_argument("--mlpq", type=str)
    data.add_argument("--monument", type=str)

    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--entities", type=str, required=True)
    parser.add_argument("--properties", type=str, required=True)
    parser.add_argument("--version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--skip-incomplete", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


Sample = collections.namedtuple("Sample", ["query", "sparql"])


SPLIT_RENAME = {
    "train": "train",
    "test": "test",
    "dev": "val",
    "valid": "val",
    "validation": "val",
}


def load_data(args: argparse.Namespace) -> tuple[str, dict[str, list[Sample]]]:
    output = {}
    if args.wikidata_simple_questions is not None:
        kg = "wikidata"
        data = load_dataset(args.wikidata_simple_questions)
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                query = item["question"]
                subj = item["answer"]["subject"]
                obj = item["answer"]["object"]
                prop = item["answer"]["predicate"]

                if prop.startswith("R"):
                    subj, obj = obj, subj
                    subj = "x"
                    prop = "P" + prop[1:]
                else:
                    obj = "x"
                prop = "wdt:" + prop

                if subj == "x":
                    subj = "?" + subj
                    obj = "wd:" + obj
                else:
                    obj = "?" + obj
                    subj = "wd:" + subj

                sparql = f"SELECT ?x WHERE {{ {subj} {prop} {obj} . }}"
                samples.append(Sample(query, sparql))
            output[split] = samples

    elif args.qald_10 is not None:
        kg = "wikidata"
        data = load_dataset(args.qald_10)
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                sparql = item["query"]["sparql"]
                queries = [
                    q["string"]
                    for q in json.loads(item["question"])
                    if q["language"] == "en"
                ]
                # replace entities and properties
                sparql = re.sub(
                    r"<http://www.wikidata.org/entity/(Q\d+?)>",
                    lambda match: "wd:" + match.group(1),
                    sparql
                )

                def _rep_prop(m: re.Match) -> str:
                    pfx = m.group(1)
                    if pfx == "direct":
                        pfx = "wdt"
                    else:
                        raise RuntimeError(f"unknown prefix {pfx}")
                    return f"{pfx}:{m.group(2)}"

                sparql = re.sub(
                    r"<http://www.wikidata.org/prop/(?:(\S+?)/)?(P\d+?)>",
                    _rep_prop,
                    sparql
                )
                for q in queries:
                    samples.append(Sample(q, sparql))

            output[split] = samples

    elif args.lc_quad2_wikidata is not None:
        kg = "wikidata"
        data = load_dataset(args.lc_quad2_wikidata, "lcquad2-wikidata")
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                queries = [item["question"]]
                sparql = item["sparql"]
                for pq in item["paraphrased_question"]:
                    queries.append(pq)
                for q in queries:
                    if q is None or q.strip() == "" or "{" in q or "}" in q:
                        continue
                    samples.append(Sample(q, sparql))
            output[split] = samples

    elif args.mcwq is not None:
        kg = "wikidata"
        with open(os.path.join(args.mcwq, "dataset.json"), "r") as inf:
            train_data = json.load(inf)
        with open(os.path.join(args.mcwq, "gold_test.json"), "r") as inf:
            test_data = json.load(inf)
        for data, split in [(train_data, "train"), (test_data, "test")]:
            samples = []
            for item in data:
                query = item["questionWithBrackets"]
                # sub out brackets
                query = re.sub(
                    r"\[(.+?)\]",
                    lambda m: m.group(1),
                    query
                )
                # repair some whitespace issues
                # words followed by 's
                query = re.sub(
                    r"(\w+)\s+('s)(?:\s+|$)",
                    lambda m: m.group(1) + m.group(2) + " ",
                    query
                )
                # punctuation with surrounding spaces
                query = re.sub(
                    r"\s+([,.?!;])(?:\s+|$)",
                    lambda m: m.group(1) + " ",
                    query
                )
                sparql = item["sparql"]
                samples.append(Sample(query, sparql))
            output[split] = samples

    elif args.qa_wiki is not None:
        kg = "wikidata"
        samples = []
        with open(args.qa_wiki, "r") as inf:
            for line in inf:
                line = line.strip()
                sparql, query = line.split("\t")
                samples.append(Sample(query, sparql))
        output["train"] = samples

    else:
        raise RuntimeError("unknown dataset")

    return kg, output


def format_query(
    query: str,
    version: str,
    kg: str
) -> str:
    if version == "v1":
        return f"Generate a SPARQL query over {kg.capitalize()} for " \
            f"the question \"{clean(query)}\""

    return preprocess_natural_language_query(query, [kg], None)


def prepare(args: argparse.Namespace):
    kg, data = load_data(args)
    parser = load_sparql_parser([kg])

    prefixes = general_prefixes()

    ent_index = KgIndex.load(
        args.entities,
        args.progress
    )
    prefixes.update(ent_index.prefixes)
    ent_indices = {kg: ent_index}

    prop_index = KgIndex.load(
        args.properties,
        args.progress
    )
    prefixes.update(prop_index.prefixes)
    prop_indices = {kg: prop_index}

    os.makedirs(args.output, exist_ok=True)

    for split, samples in data.items():
        input = os.path.join(
            args.output,
            f"{split}_input.txt"
        )
        input_raw = os.path.join(
            args.output,
            f"{split}_input.raw.txt"
        )
        assert len(samples) > 0, f"no samples for split {split}"
        target = os.path.join(
            args.output,
            f"{split}_sparql.txt"
        )
        target_raw = os.path.join(
            args.output,
            f"{split}_sparql.raw.txt"
        )
        examples = os.path.join(
            args.output,
            f"{split}_examples.tsv"
        )
        incomplete = 0
        invalid = 0
        total = 0
        with open(input, "w") as inf, \
                open(input_raw, "w") as inrf, \
                open(target, "w") as tf, \
                open(target_raw, "w") as trf, \
                open(examples, "w") as ef:
            for sample in tqdm(
                samples,
                desc=f"processing and writing {split} samples",
                leave=False,
                disable=not args.progress
            ):
                # clean sparql in sample
                sample = Sample(
                    sample.query,
                    clean(sample.sparql),
                )

                try:
                    sparqls, inc = replace_iris(
                        sample.sparql,
                        parser,
                        ent_indices,
                        prop_indices,
                        args.version,
                        "in_order" if split == "train" else "only_first"
                    )
                except Exception:
                    invalid += 1
                    continue

                incomplete += inc
                if args.skip_incomplete and inc:
                    continue

                total += len(sparqls)
                if len(sparqls) == 0:
                    continue

                # same as above, but without replacing
                # with natural language entities
                raw_sparql = fix_prefixes(
                    sample.sparql,
                    parser,
                    prefixes,
                )
                raw_sparql = replace_vars_and_special_tokens(
                    raw_sparql,
                    parser,
                    args.version
                )

                ef.write(json.dumps({
                    "query": sample.query,
                    "sparql": sparqls[0]
                }) + "\n")

                for sparql in sparqls:
                    sparql = fix_prefixes(
                        sparql,
                        parser,
                        prefixes,
                    )
                    sparql = replace_vars_and_special_tokens(
                        sparql,
                        parser,
                        args.version
                    )
                    tf.write(json.dumps(sparql) + "\n")
                    trf.write(raw_sparql + "\n")
                    inf.write(
                        json.dumps([{
                            "role": "user",
                            "text": format_query(
                                sample.query,
                                args.version,
                                kg
                            )}]) + "\n"
                    )
                    inrf.write(json.dumps(sample.query) + "\n")

        print(
            f"Generated {total:,} SPARQL queries while "
            f"processing {len(samples):,} {split} samples with "
            f"{incomplete:,} ({incomplete / len(samples):.2%}) "
            f"being incomplete and {invalid:,} ({invalid / len(samples):.2%}) "
            "being invalid"
        )


if __name__ == "__main__":
    prepare(parse_args())
