import argparse
import sys
import logging
import random
import os
import re
import json
import string
import multiprocessing as mp
from dataclasses import dataclass

from search_index import normalize
from tqdm import tqdm
from datasets import load_dataset

from sparql_kgqa.sparql.utils import find_all
from sparql_kgqa.sparql.utils2 import (
    SEARCH_TOKEN,
    OBJ_TYPES,
    Chat,
    KgManager,
    clean,
    extract_fields,
    load_kg_manager,
    partition_by,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    data = parser.add_mutually_exclusive_group(required=True)

    # wikidata
    data.add_argument("--wikidata-simple-questions", action="store_true")
    data.add_argument("--lc-quad2-wikidata", action="store_true")
    data.add_argument("--qald-10", action="store_true")
    data.add_argument("--qald-7", type=str)
    data.add_argument("--mcwq", type=str)
    data.add_argument("--wwq", type=str)
    data.add_argument("--kqa-pro", type=str)
    data.add_argument("--qa-wiki", type=str)
    data.add_argument("--qlever-wikidata", type=str)
    data.add_argument("--instruct-to-sparql", action="store_true")
    # data.add_argument("--time-questions", type=str)
    # data.add_argument("--cron-questions", type=str)
    # data.add_argument("--mkqa", type=str)
    # data.add_argument("--mintaka", type=str)

    # freebase
    data.add_argument("--grail-qa", action="store_true")
    data.add_argument("--wqsp", action="store_true")
    data.add_argument("--cwq", action="store_true")
    data.add_argument("--cfq", type=str)
    # data.add_argument("--freebase-simple-questions", type=str)
    # data.add_argument("--30mqa", type=str)
    # data.add_argument("--graph-questions", type=str)

    # dbpedia
    data.add_argument("--lc-quad1-dbpedia", action="store_true")
    data.add_argument("--qald-9", action="store_true")
    # data.add_argument("--simple-dbpedia-qa", type=str)
    # data.add_argument("--mlpq", type=str)
    # data.add_argument("--monument", type=str)

    # dblp
    data.add_argument("--dblp-quad", action="store_true")

    # orkg
    data.add_argument("--sci-qa", action="store_true")

    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--entities", type=str)
    parser.add_argument("--properties", type=str)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--skip", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--qlever-endpoint", type=str, default=None)
    sample_group = parser.add_argument_group("samples")
    sample_group.add_argument("--max-samples", type=int, default=None)
    sample_group.add_argument("--stages-per-sample", type=int, default=None)
    sample_group.add_argument("--selections-min-k", type=int, default=5)
    sample_group.add_argument("--selections-max-k", type=int, default=5)
    sample_group.add_argument("--skip-stages", action="store_true")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("-n", "--num-workers", type=int, default=None)
    parser.add_argument(
        "--index-types", nargs="+", default=["prefix"], choices=["prefix", "qgram"]
    )
    return parser.parse_args()


@dataclass
class Sample:
    question: str
    sparql: str
    info: dict | None = None
    invalid: bool = False


SPLIT_RENAME = {
    "train": "train",
    "test": "test",
    "dev": "val",
    "valid": "val",
    "validation": "val",
}

COMMON_SPARQL_KEYWORDS = [
    "PREFIX",
    "SELECT",
    "DISTINCT",
    "WHERE",
    "FILTER",
    "ORDER",
    "LIMIT",
    "OFFSET",
    "OPTIONAL",
    "UNION",
    "GROUP",
    "HAVING",
    "VALUES",
]


def clean_sparql_for_wqsp_and_cwq(sparql: str) -> str:
    lines = []
    for line in sparql.splitlines():
        comment = line.find("#")
        if comment != -1:
            line = line[:comment]
        line = line.replace(" OR ", " || ")
        lines.append(line)

    return "\n".join(line for line in lines if line.strip())


def load_data(args: argparse.Namespace) -> tuple[str, dict[str, list[Sample]]]:
    output = {}
    if args.wikidata_simple_questions:
        kg = "wikidata"
        data = load_dataset("third_party/KGQA-datasets/simple_wikidata_qa")
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

    elif args.lc_quad2_wikidata:
        kg = "wikidata"
        data = load_dataset("third_party/KGQA-datasets/lcquad_v2", "lcquad2-wikidata")
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                queries = [item["question"]]
                sparql = item["sparql"]
                if split != "test":
                    queries.extend(item["paraphrased_question"])
                for q in queries:
                    if q is None or q.strip() == "" or "{" in q or "}" in q:
                        continue
                    samples.append(Sample(q, sparql))
            output[split] = samples

    elif args.qald_10:
        kg = "wikidata"
        data = load_dataset("third_party/KGQA-datasets/qald/qald-10.py")
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

                for q in queries:
                    samples.append(Sample(q, sparql))
                    if split == "test":
                        # only the first for test
                        break

            output[split] = samples

    elif args.qald_7 is not None:
        kg = "wikidata"
        with open(
            os.path.join(args.qald_7, "qald-7-train-en-wikidata.json"), "r"
        ) as inf:
            train = json.load(inf)

        with open(
            os.path.join(args.qald_7, "qald-7-test-en-wikidata.json"), "r"
        ) as inf:
            test = json.load(inf)

        for data, split in [(train, "train"), (test, "test")]:
            samples = []
            for item in data["questions"]:
                for q in item["question"]:
                    if q["language"] != "en":
                        continue
                    sparql = item["query"]["sparql"]
                    samples.append(Sample(q["string"], sparql))
                    if split == "test":
                        break
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
                query = re.sub(r"\[(.+?)\]", lambda m: m.group(1), query)
                # repair some whitespace issues
                # words followed by 's
                query = re.sub(
                    r"(\w+)\s+('s)(?:\s+|$)",
                    lambda m: m.group(1) + m.group(2) + " ",
                    query,
                )
                # punctuation with surrounding spaces
                query = re.sub(
                    r"\s+([,.?!;])(?:\s+|$)", lambda m: m.group(1) + " ", query
                )
                sparql = item["sparql"]
                samples.append(Sample(query, sparql))
            output[split] = samples

    elif args.wwq is not None:
        kg = "wikidata"
        for split in ["train", "dev", "test"]:
            file = os.path.join(args.wwq, f"{split}.json")
            split = SPLIT_RENAME.get(split, split)
            with open(file, "r") as inf:
                data = json.load(inf)

            samples = []
            for item in data:
                query = item["utterance"]
                sparql = item["sparql"]
                info = extract_fields(item, ["id", "results"])
                samples.append(Sample(query, sparql, info))
            output[split] = samples

    elif args.kqa_pro is not None:
        raise NotImplementedError
        # kg = "wikidata"
        # for split in ["train", "val", "test"]:
        #     file = os.path.join(args.kqa_pro, f"{split}.json")
        #     with open(file, "r") as inf:
        #         data = json.load(inf)
        #
        #     for item in data:
        #         query = item["question"]
        #         sparql = item.get("sparql", "")
        #         samples.append(Sample(query, sparql))
        #     output[split] = samples

    elif args.qa_wiki is not None:
        kg = "wikidata"
        samples = []
        with open(args.qa_wiki, "r") as inf:
            for line in inf:
                line = line.strip()
                sparql, query = line.split("\t")
                samples.append(Sample(query, sparql))
        output["train"] = samples

    elif args.qlever_wikidata is not None:
        kg = "wikidata"
        samples = []
        with open(args.qlever_wikidata, "r") as inf:
            for line in inf:
                line = line.strip()
                query, sparql = line.split("\t")
                samples.append(Sample(query, sparql))
        output["train"] = samples

    elif args.instruct_to_sparql:
        kg = "wikidata"
        full = load_dataset("PaDaS-Lab/Instruct-to-SPARQL", split="full")
        full_ids = set(item["id"] for item in full)
        split_ids = set()
        for split in ["train", "validation", "test"]:
            items = load_dataset("PaDaS-Lab/Instruct-to-SPARQL", split=split)
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                split_ids.add(item["id"])
                for q in item["instructions"]:
                    samples.append(Sample(q, item["sparql_query"]))
                    if split == "test":
                        break

            output[split] = samples

        not_found = full_ids - split_ids
        for item in full:
            if item["id"] not in not_found:
                continue

            # add not found items to train
            # not found usually means the sparql timed out
            # during execution
            for q in item["instructions"]:
                output["train"].append(Sample(q, item["sparql_query"]))

    elif args.grail_qa:
        kg = "freebase"
        data = load_dataset("third_party/KGQA-datasets/grail_qa", "grail_qa")

        output["train"] = [
            Sample(item["question"], item["sparql_query"]) for item in data["train"]
        ]
        output["val"] = [
            Sample(item["question"], item["sparql_query"])
            for item in data["validation"]
        ]

        data = load_dataset("third_party/KGQA-datasets/grail_qa", "grailqa_test_public")
        output["test"] = [Sample(item["question"], "") for item in data["test"]]

    elif args.wqsp:
        data = load_dataset("third_party/KGQA-datasets/webqsp")
        kg = "freebase"
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                for sparql in item["Parses"]["Sparql"]:
                    samples.append(
                        Sample(
                            item["RawQuestion"], clean_sparql_for_wqsp_and_cwq(sparql)
                        )
                    )
                    if split == "test":
                        # only first for test
                        break
            output[split] = samples

    elif args.cwq:
        data = load_dataset(
            "third_party/KGQA-datasets/complex_web_questions", "complex_web_questions"
        )
        kg = "freebase"
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                samples.append(
                    Sample(
                        item["question"], clean_sparql_for_wqsp_and_cwq(item["sparql"])
                    )
                )
            output[split] = samples

        data = load_dataset(
            "third_party/KGQA-datasets/complex_web_questions",
            "complexwebquestions_test",
        )
        output["test"] = [
            Sample(item["question"], clean_sparql_for_wqsp_and_cwq(item["sparql"]))
            for item in data["test"]
        ]

    elif args.cfq is not None:
        kg = "freebase"
        split = os.path.join(args.cfq, "splits", "random_split.json")
        dataset = os.path.join(args.cfq, "dataset.json")
        with open(split, "r") as inf:
            split = json.load(inf)

        with open(dataset, "r") as inf:
            data = json.load(inf)

        for s in ["train", "dev", "test"]:
            indices = split[f"{s}Idxs"]
            s = SPLIT_RENAME.get(s, s)
            samples = []
            for idx in indices:
                item = data[idx]
                samples.append(Sample(item["question"], item["sparql"]))
            output[s] = samples

    elif args.lc_quad1_dbpedia:
        kg = "dbpedia"
        data = load_dataset("third_party/KGQA-datasets/lcquad_v1", "lcquad")
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                samples.append(
                    Sample(
                        item["corrected_question"].replace(" ?", "?"),
                        item["sparql_query"],
                    )
                )
            output[split] = samples

    elif args.qald_9:
        kg = "dbpedia"
        data = load_dataset("third_party/KGQA-datasets/qald/qald-9.py")
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

                for q in queries:
                    samples.append(Sample(q, sparql))
                    if split == "test":
                        # only first for test
                        break

            output[split] = samples

    elif args.dblp_quad:
        kg = "dblp"
        data = load_dataset("awalesushil/DBLP-QuAD")
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                samples.append(
                    Sample(item["question"]["string"], item["query"]["sparql"])
                )
                if split == "test":
                    continue
                samples.append(
                    Sample(
                        item["paraphrased_question"]["string"], item["query"]["sparql"]
                    )
                )
            output[split] = samples

    elif args.dblp_quad:
        kg = "orkg"
        data = load_dataset("orkg/SciQA")
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                samples.append(
                    Sample(item["question"]["string"], item["query"]["sparql"])
                )
                if split == "test":
                    continue

                for q in item["paraphrased_question"]:
                    samples.append(Sample(q, item["query"]["sparql"]))

            output[split] = samples

    else:
        raise RuntimeError("unknown dataset")

    return kg, output


# nltk english stopwords
STOP = set(
    [
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "you're",
        "you've",
        "you'll",
        "you'd",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "she's",
        "her",
        "hers",
        "herself",
        "it",
        "it's",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "that'll",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "don't",
        "should",
        "should've",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "aren't",
        "couldn",
        "couldn't",
        "didn",
        "didn't",
        "doesn",
        "doesn't",
        "hadn",
        "hadn't",
        "hasn",
        "hasn't",
        "haven",
        "haven't",
        "isn",
        "isn't",
        "ma",
        "mightn",
        "mightn't",
        "mustn",
        "mustn't",
        "needn",
        "needn't",
        "shan",
        "shan't",
        "shouldn",
        "shouldn't",
        "wasn",
        "wasn't",
        "weren",
        "weren't",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
    ]
)

PUNCT = set(string.punctuation)


def get_search_query(question: str, name: str, index_type: str) -> str:
    keywords = normalize(name).split()

    if index_type == "prefix":
        # special handling of prefix index
        keywords = list(
            k
            for k in set(keywords)
            if k not in STOP
            and k not in PUNCT
            and not any(k.startswith(other) and k != other for other in keywords)
        )
        query_keywords = normalize(question).split()
        matching, non_matching = partition_by(
            keywords, lambda kw: any(qk.startswith(kw) for qk in query_keywords)
        )
        random.shuffle(matching)
        random.shuffle(non_matching)
        keywords = matching + non_matching

    # assuming more important keywords are at the beginning
    # limit the number of keywords to 3-5
    end = random.randint(3, max(3, min(len(keywords), 5)))
    keywords = keywords[:end]

    return " ".join(keywords)


def is_lit_or_other(obj_type: str) -> bool:
    return obj_type in ["literal", "other"]


def is_ent_or_prop(obj_type: str) -> bool:
    return obj_type in ["entity", "property"]


def prepare_stages(
    question: str, sparql: str, managers: list[KgManager], args: argparse.Namespace
) -> list[tuple[str | Chat, str]]:
    manager = random.choice(managers)
    sparql_encoded = sparql.encode()
    parse = manager.parser.parse(sparql, skip_empty=True, collapse_single=False)

    mappings = {"entity": manager.entity_mapping, "property": manager.property_mapping}
    indices = {"entity": manager.entity_index, "property": manager.property_index}

    def span(obj, start=sys.maxsize, end=0) -> tuple[int, int]:
        if "children" in obj:
            for child in obj["children"]:
                start, end = span(child, start, end)
            return start, end

        f, t = obj["byte_span"]
        return min(start, f), max(end, t)

    def map_item(obj):
        # return tuple with identifier, variant, label, synonyms
        # and additional info
        if obj["name"] in ["RDFLiteral", "NumericLiteral"]:
            child = obj["children"][0]["children"][0]
            label = child["value"].strip("'").strip('"')
            return "literal", label, None, label, []

        if obj["name"] == "BooleanLiteral":
            label = obj["children"][0]["children"][0]
            return "literal", label, None, label, []

        if obj["name"] != "iri":
            return None

        child = obj["children"][0]
        if child["name"] == "PrefixedName":
            pfx, val = child["children"][0]["value"].split(":", 1)
            long = manager.prefixes.get(pfx, manager.custom_prefixes.get(pfx))
            if long is None:
                return None

            iri = long + val + ">"

        elif child["name"] == "IRIREF":
            iri = child["value"]

        else:
            return None

        # check whether the iri is a valid entity or property
        matching = {}
        for obj_type in ["entity", "property"]:
            map = mappings[obj_type]
            index = indices[obj_type]
            norm = map.normalize(iri)
            if norm is None or norm[0] not in map:
                continue

            id = map[norm[0]]
            label = index.get_name(id)
            syns = [s for s in index.get_val(id, 2).split(";;;") if s != ""]
            matching[obj_type] = (obj_type, *norm, label, syns)

        if matching:
            return matching.get("property", None) or matching.get("entity", None)

        elif manager.find_longest_prefix(iri, manager.prefixes) is not None:
            return "other", iri, None, iri, []

    # get all items in triples
    items = [
        (item, map_item(item))
        for item in find_all(parse, name="iri", skip={"Prologue"})
    ] + [
        # only literals in triples are searchable
        # rest should be predicted directly
        (item, map_item(item))
        for triple in find_all(parse, name="TriplesSameSubject")
        for item in find_all(
            triple,
            name={"RDFLiteral", "NumericLiteral", "BooleanLiteral"},
        )
    ]

    # filter out invalid items and sort by occurence in the query
    items = sorted(
        ((item, processed) for item, processed in items if processed is not None),
        key=lambda x: span(x[0]),
    )

    samples = []
    for item_idx in random.sample(
        list(range(len(items) + 1)),
        min(len(items) + 1, args.stages_per_sample or (len(items) + 1)),
    ):
        manager = random.choice(managers)

        if item_idx >= len(items):
            if item_idx == 0:
                end = 0
            else:
                end = span(items[item_idx - 1][0])[1]

            sparql_prefix = manager.fix_prefixes(
                sparql_encoded[:end].decode(), is_prefix=True, remove_known=True
            )
            natural_prefix, selections, inc = manager.replace_iris(
                sparql_prefix, is_prefix=True
            )
            if inc:
                continue

            final_cont = sparql_encoded[end:].decode()
            final_cont_prompt = manager.get_sparql_continuation_prompt(
                question, sparql_prefix, natural_prefix, selections
            )
            samples.append((final_cont_prompt, final_cont))
            continue

        item, processed = items[item_idx]
        start, end = span(item)
        assert end >= start, "invalid span"
        obj_type, identifier, variant, label, syns = processed

        sparql_prefix = manager.fix_prefixes(
            sparql_encoded[:start].decode(),
            is_prefix=True,
            remove_known=True,
        )
        natural_prefix, selections, inc = manager.replace_iris(
            sparql_prefix, is_prefix=True
        )
        if inc:
            continue

        # 1. randomly drop items with type other or literal
        # such that some continuations are trained to
        # predict them directly rather than searching for them
        # 2. randomly drop entities or properties that already
        # occured in the previous triples

        if item_idx > 0:
            last_end = span(items[item_idx - 1][0])[1]
            assert last_end < start, "invalid item order"
            last_sparql_prefix = manager.fix_prefixes(
                sparql_encoded[:last_end].decode(), is_prefix=True, remove_known=True
            )
            last_natural_prefix, last_selection, inc = manager.replace_iris(
                last_sparql_prefix, is_prefix=True
            )
            if inc:
                continue

        else:
            last_sparql_prefix = ""
            last_natural_prefix = ""
            last_selection = []
            last_end = 0

        def is_known_ent_or_prop(
            obj_type: str,
            identifier: str,
        ) -> bool:
            return is_ent_or_prop(obj_type) and any(
                items[i][1][1] == identifier for i in range(item_idx)
            )

        start_idx = item_idx
        while start_idx < len(items):
            _, (start_obj_type, start_identifier, *_) = items[start_idx]
            if random.random() < 0.5:
                break

            is_other = is_lit_or_other(start_obj_type)
            is_known = is_known_ent_or_prop(start_obj_type, start_identifier)
            if is_other or is_known:
                start_idx += 1
            else:
                break

        if start_idx < len(items) - 1:
            start, _ = span(items[item_idx + 1][0])
            token = SEARCH_TOKEN
        else:
            start = len(sparql_encoded)
            token = ""

        cont = sparql_encoded[last_end:start].decode() + token

        cont_prompt = manager.get_sparql_continuation_prompt(
            question, last_sparql_prefix, last_natural_prefix, last_selection
        )

        samples.append((cont_prompt, cont))

        if len(syns) > 0:
            num_search_failures = min(
                random.sample(
                    list(range(len(syns) + 1)),
                    1,
                    counts=list(range(len(syns) + 1, 0, -1)),
                )[0],
                3,
            )
            search_failures = set(
                get_search_query(question, syn, manager.entity_index.get_type())
                for syn in random.sample(syns, num_search_failures)
            )
        else:
            search_failures = set()

        search = get_search_query(question, label, manager.entity_index.get_type())
        if len(search_failures) >= len(syns) and random.random() < 0.5:
            search_failures.add(search)
            search = random.choice(list(search_failures))

        search_prompt, _ = manager.get_search_prompt_and_regex(
            question, natural_prefix, selections, search_failures
        )
        samples.append((search_prompt, search))

        selection_k = random.randint(args.selections_min_k, args.selections_max_k)

        alts = manager.get_selection_alternatives(
            sparql_prefix,
            search,
            selection_k,
            max_candidates=16384,
            endpoint=args.qlever_endpoint,
            max_retries=3,
        )

        target_alts = [
            (obj_type, alt)
            for obj_type, obj_alts in alts.items()
            for alt in obj_alts
            if (alt.identifier == identifier or alt.short_identifier == identifier)
            and (variant is None or variant in (alt.variants or {}))
        ]
        target_obj_type, target_alt = (
            random.choice(target_alts) if target_alts else (obj_type, None)
        )

        select_failures = set()

        if (
            target_alt is not None
            and random.random() < 0.1
            and len(alts[target_obj_type]) > 1
        ):
            # differentiate between dropping the target from
            # the list of alternatives entirely or only adding the
            # variant to previous fails
            if random.random() < 0.5:
                alts[target_obj_type].remove(target_alt)
            else:
                select_failures.add((target_obj_type, identifier, variant))

            target_alt = None

        alts_to_fail = [
            (target_obj_type, alt.identifier, var)
            for alt in alts.get(target_obj_type, [])
            for var in (alt.variants or [None])
            if target_alt is None
            or alt.identifier != target_alt.identifier
            or var != variant
        ]

        if len(alts_to_fail) > 0:
            num_select_failures = min(
                random.sample(
                    list(range(len(alts_to_fail) + 1)),
                    1,
                    counts=list(range(len(alts_to_fail) + 1, 0, -1)),
                )[0],
                3 - len(select_failures),
            )
            select_failures.update(
                random.sample(
                    alts_to_fail,
                    num_select_failures,
                    # make earlier alts more likely failures
                    counts=list(range(len(alts_to_fail), 0, -1)),
                )
            )

        select_prompt, _ = manager.get_selection_prompt_and_regex(
            question,
            natural_prefix,
            search,
            alts,
            selections=selections,
            failures=select_failures,
        )
        if target_alt is None:
            selection = "0. none"
        else:
            offset = sum(
                len(alts[obj_type])
                for obj_type in OBJ_TYPES[: OBJ_TYPES.index(target_obj_type)]
                if obj_type in alts
            )
            idx = offset + alts[target_obj_type].index(target_alt)
            selection = f"{idx + 1}. {target_alt.get_sparql_label(variant)}"

        samples.append((select_prompt, selection))

    return samples


def prepare_sample(
    sample: Sample, args: argparse.Namespace, split: str
) -> tuple[Sample, list[tuple[str | Chat, str]]]:
    global managers

    # clean sparql in sample
    sample.question = clean(sample.question)
    sample.sparql = clean(sample.sparql)
    manager = random.choice(managers)

    try:
        sample.sparql = manager.fix_prefixes(sample.sparql)
    except Exception:
        sample.invalid = True

    if split == "test" or sample.invalid or args.skip_stages:
        return sample, []

    stages = prepare_stages(sample.question, sample.sparql, managers, args)
    return sample, stages


def prepare_sample_mp(args: tuple[Sample, argparse.Namespace, str]):
    return prepare_sample(*args)


managers: list[KgManager] = []


def init(kg: str, args: argparse.Namespace):
    global managers
    random.seed(args.seed)

    for index_type in args.index_types:
        manager = load_kg_manager(
            kg,
            args.entities,
            args.properties,
            index_type,
        )
        managers.append(manager)


def prepare(args: argparse.Namespace):
    random.seed(args.seed)
    logging.basicConfig(
        format="[%(asctime)s] {%(name)s - %(levelname)s} %(message)s",
        level=logging.INFO,
    )
    kg, data = load_data(args)
    num_samples = {s: len(samples) for s, samples in data.items()}
    print(f"Number of raw samples: {num_samples}")

    os.makedirs(args.output, exist_ok=True)

    if not args.sequential:
        print(f"Starting {args.num_workers} workers")
        pool = mp.Pool(args.num_workers, initializer=init, initargs=(kg, args))
    else:
        pool = None
        init(kg, args)

    if (args.stages_per_sample or 1) < 0:
        train_samples = num_samples["train"]
        # determine dynamically based on the number of training samples
        # < 1000: 8x stages
        # < 10000: 4x stages
        # < 100000: 2x stages
        # >= 100000: 1x stage
        if train_samples < 1000:
            args.stages_per_sample = 8
        elif train_samples < 10000:
            args.stages_per_sample = 4
        elif train_samples < 100000:
            args.stages_per_sample = 2
        else:
            args.stages_per_sample = 1

    for split, samples in data.items():
        if split in args.skip:
            print(f"skipping {split} split")
            continue

        if args.max_samples is not None:
            samples = random.sample(samples, min(len(samples), args.max_samples))

        input = os.path.join(args.output, f"{split}_input.jsonl")
        if os.path.exists(input) and not args.overwrite:
            print(f"skipping {split} split because it already exists")
            continue

        target = os.path.join(args.output, f"{split}_target.jsonl")
        raw = os.path.join(args.output, f"{split}_raw.jsonl")
        outputs = list(
            tqdm(
                (
                    (prepare_sample(sample, args, split) for sample in samples)
                    if pool is None
                    else pool.imap(
                        prepare_sample_mp, ((sample, args, split) for sample in samples)
                    )
                ),  # type: ignore
                desc=f"processing and writing {split} samples",
                leave=False,
                total=len(samples),
                disable=not args.progress,
            )
        )
        random.shuffle(outputs)

        invalid = 0
        num_sparqls = 0
        with open(input, "w") as inf, open(target, "w") as tf, open(raw, "w") as rf:
            for output in outputs:
                (sample, stages) = output

                invalid += int(sample.invalid)

                rf.write(
                    json.dumps(
                        {
                            "question": sample.question,
                            "sparql": sample.sparql,
                            "info": sample.info or {},
                        }
                    )
                    + "\n"
                )

                if split == "test":
                    assert len(stages) == 0
                    inf.write(json.dumps(sample.question) + "\n")
                    tf.write(json.dumps(sample.sparql) + "\n")
                    continue

                num_sparqls += len(stages)
                for prompt, target in stages:
                    if isinstance(prompt, str):
                        prompt = [{"role": "user", "text": prompt}]
                    inf.write(json.dumps(prompt) + "\n")
                    tf.write(json.dumps(target) + "\n")

        print(
            f"Processed {len(samples):,} {split} samples with "
            f"{invalid:,} ({invalid / len(samples):.2%}) "
            f"being incomplete or invalid.\n"
            f"Generated {num_sparqls:,} additional samples."
        )


if __name__ == "__main__":
    prepare(parse_args())
