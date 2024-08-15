import argparse
import random
import os
import re
import json
import collections
from typing import Any

from search_index import PrefixIndex, QGramIndex
from tqdm import tqdm
from datasets import load_dataset

from sparql_kgqa.sparql.utils import find_all, parse_to_string
from sparql_kgqa.sparql.utils2 import (
    KgManager,
    Mapping,
    WikidataManager,
    WikidataPropertyMapping,
    clean,
    run_parallel
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
    parser.add_argument("--entities-index", type=str, default="qgram", choices=[
        "qgram", "prefix"
    ])
    parser.add_argument("--properties", type=str, required=True)
    parser.add_argument("--properties-index", type=str, default="qgram", choices=[
        "qgram", "prefix"
    ])
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--samples-per-sample", type=int, default=1)
    selection_group = parser.add_argument_group("selection")
    selection_group.add_argument("--with-selections", action="store_true")
    selection_group.add_argument(
        "--selections-per-sample",
        type=int,
        default=1
    )
    selection_group.add_argument(
        "--selections-min-k",
        type=int,
        default=4
    )
    selection_group.add_argument(
        "--selections-max-k",
        type=int,
        default=32
    )
    selection_group.add_argument(
        "--selections-min-aliases",
        type=int,
        default=0
    )
    selection_group.add_argument(
        "--selections-max-aliases",
        type=int,
        default=8
    )
    selection_group.add_argument(
        "--selections-dropout",
        type=float,
        default=0.1
    )
    selection_group.add_argument(
        "--selections-add-info-p",
        type=float,
        default=0.1
    )
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("-n", "--num-workers", type=int, default=None)
    return parser.parse_args()


Sample = collections.namedtuple("Sample", ["question", "sparql"])


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


def prepare_selection(
    question: str,
    raw_sparql: str,
    manager: KgManager,
    args: argparse.Namespace
) -> list[tuple[str, str]]:
    selections = []
    for _ in range(args.selections_per_sample):
        parse = manager.parser.parse(
            raw_sparql,
            skip_empty=True,
            collapse_single=False
        )

        def map_item(obj) -> tuple[Any, str, str, int] | None:
            if obj["name"] != "iri":
                return None
            child = obj["children"][0]
            if child["name"] != "PrefixedName":
                return None
            child = child["children"][0]
            if child["name"] != "PNAME_LN":
                return None
            pfx, val = child["value"].split(":", 1)
            return obj, pfx, val, child["byte_span"][0]

        items = list(filter(
            lambda item: item is not None
            and item[1] in manager.custom_prefixes,
            map(
                map_item,
                find_all(
                    parse,
                    name="iri",
                    skip={"Prologue"}
                )
            )
        ))
        if len(items) == 0:
            continue

        item = random.choice(items)
        assert item is not None
        item, pfx, iri, start = item
        long = manager.custom_prefixes[pfx]
        iri = long + iri + ">"

        raw_encoded = raw_sparql.encode()
        prefix_raw = raw_encoded[:start].decode(errors="replace")
        assert not manager.replace_iri(item, replacement="synonyms")
        replaced = parse_to_string(item)

        prefix, _ = manager.replace_iris(
            prefix_raw,
            replacement="synonyms",
            is_prefix=True
        )
        result = manager.get_selection_alternatives(
            prefix_raw + replaced,
            random.randint(
                args.selections_min_k,
                args.selections_max_k
            )
        )
        if result is None:
            continue

        alts, obj_type, guess = result

        if obj_type == "entity":
            mapping = manager.entity_mapping
        else:
            mapping = manager.property_mapping

        norm = mapping.normalize(iri)
        if norm is None:
            continue

        # variant dropout
        default_variants = mapping.default_variants()
        if (
            guess[1] is not None
            and default_variants
            and random.random() < args.selections_dropout
        ):
            random_variant = random.choice(
                list(default_variants) + [None]
            )
            guess = (guess[0], random_variant)

        # alternative dropout
        identifier, variant = norm
        i = 0
        for alt in alts:
            if alt.identifier == identifier:
                break
            i += 1

        if i < len(alts) and random.random() < args.selections_dropout:
            alts.pop(i)
            i = len(alts)

        if i == len(alts):
            target_alt = f"{len(alts) + 1}. none"
        else:
            target_alt = f"{i + 1}. {alts[i].label}"
            if variant is not None:
                target_alt += f" ({variant})"

        prefix = manager.fix_prefixes(prefix, is_prefix=True)

        prompt, _ = manager.get_selection_prompt_and_regex(
            question,
            prefix,
            obj_type,
            guess,
            alts,
            max_aliases=random.randint(
                args.selections_min_aliases,
                args.selections_max_aliases
            ),
            add_infos=random.random() < args.selections_add_info_p
        )
        selections.append((prompt, target_alt))

    return selections


def prepare_sample(
    sample: Sample,
    manager: KgManager,
    args: argparse.Namespace,
    split: str
) -> tuple[
    str,
    str,
    str,
    list[str],
    list[tuple[str, str]],
] | None:
    # clean sparql in sample
    sample = Sample(
        sample.question,
        clean(sample.sparql),
    )

    sparqls = []
    for _ in range(max(1, args.samples_per_sample)):
        try:
            sparql, inc = manager.replace_iris(
                sample.sparql,
                replacement="label" if split == "test" else "synonyms"
            )
            sparql = manager.fix_prefixes(sparql)
        except Exception:
            break

        if inc:
            break

        sparqls.append(sparql)

        if split == "test":
            break

    if len(sparqls) == 0:
        return None

    # same as above, but without replacing
    # with natural language entities
    raw_sparql = manager.fix_prefixes(sample.sparql)

    prompt = manager.get_sparql_prompt(sample.question)

    if not args.with_selections or split == "test":
        # do not include selection samples in test split
        selections = []
    else:
        selections = prepare_selection(
            sample.question,
            raw_sparql,
            manager,
            args
        )

    return (
        sample.question,
        raw_sparql,
        prompt,
        sparqls,
        selections
    )


def prepare(args: argparse.Namespace):
    random.seed(args.seed)
    kg, data = load_data(args)

    if kg != "wikidata":
        raise RuntimeError("only wikidata supported for now")

    entities_data = os.path.join(args.entities, "data.tsv")
    entities_index = os.path.join(args.entities, args.entities_index_type)
    if args.entities_index_type == "qgram":
        ent_index = QGramIndex.load(entities_data, entities_index)
    else:
        ent_index = PrefixIndex.load(entities_data, entities_index)
    ent_mapping = Mapping.load(
        ent_index,
        os.path.join(entities_index, "index.mapping")
    )

    properties_data = os.path.join(args.properties, "data.tsv")
    properties_index = os.path.join(
        args.properties, args.properties_index_type)
    if args.properties_index_type == "qgram":
        prop_index = QGramIndex.load(properties_data, properties_index)
    else:
        prop_index = PrefixIndex.load(properties_data, properties_index)
    prop_mapping = WikidataPropertyMapping.load(
        prop_index,
        os.path.join(properties_index, "index.mapping")
    )
    assert isinstance(prop_mapping, WikidataPropertyMapping)

    manager = WikidataManager(
        ent_index,
        prop_index,
        ent_mapping,
        prop_mapping
    )

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
        invalid = 0
        num_samples = 0
        num_selections = 0
        with open(input, "w") as inf, \
                open(input_raw, "w") as inrf, \
                open(target, "w") as tf, \
                open(target_raw, "w") as trf, \
                open(examples, "w") as ef:
            for output in tqdm(
                run_parallel(
                    prepare_sample,
                    ((sample, manager, args, split) for sample in samples),
                    args.num_workers
                ),
                desc=f"processing and writing {split} samples",
                leave=False,
                total=len(samples),
                disable=not args.progress
            ):
                if output is None:
                    invalid += 1
                    continue

                (question, raw_sparql, prompt, sparqls, selections) = output
                ef.write(json.dumps({
                    "question": question,
                    "sparql": raw_sparql
                }) + "\n")

                num_samples += len(sparqls)
                for sparql in sparqls:
                    tf.write(json.dumps(sparql) + "\n")
                    trf.write(raw_sparql + "\n")
                    inf.write(
                        json.dumps([{
                            "role": "user",
                            "text": prompt
                        }]) + "\n"
                    )
                    inrf.write(json.dumps(question) + "\n")

                num_selections += len(selections)
                for prompt, target in selections:
                    inf.write(
                        json.dumps([{
                            "role": "user",
                            "text": prompt
                        }]) + "\n"
                    )
                    inrf.write(json.dumps(prompt) + "\n")
                    tf.write(json.dumps(target) + "\n")
                    trf.write(target + "\n")

        print(
            f"Generated {num_samples:,} SPARQL queries while "
            f"processing {len(samples):,} {split} samples with "
            f"{invalid:,} ({invalid / len(samples):.2%}) "
            f"being incomplete or invalid"
        )
        if args.with_selections and split != "test":
            print(
                f"Generated {num_selections:,} selection samples"
            )


if __name__ == "__main__":
    prepare(parse_args())
