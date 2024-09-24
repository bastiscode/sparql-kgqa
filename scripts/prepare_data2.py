import argparse
import sys
import logging
import random
import os
import re
import json
import collections
import multiprocessing as mp

from search_index import normalize
from tqdm import tqdm
from datasets import load_dataset

from sparql_kgqa.sparql.utils import find_all
from sparql_kgqa.sparql.utils2 import (
    SEARCH_TOKEN,
    OBJ_TYPES,
    Chat,
    KgManager,
    WikidataPropertyMapping,
    get_kg_manager,
    clean,
    load_index_and_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    data = parser.add_mutually_exclusive_group(required=True)

    # wikidata
    data.add_argument("--wikidata-simple-questions", action="store_true")
    data.add_argument("--lc-quad2-wikidata", action="store_true")
    data.add_argument("--qald-10", action="store_true")
    data.add_argument("--mcwq", type=str)
    data.add_argument("--kqa-pro", type=str)
    data.add_argument("--qa-wiki", type=str)
    data.add_argument("--qlever-wikidata", type=str)
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

    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--entities", type=str, required=True)
    parser.add_argument("--properties", type=str, required=True)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip", nargs="+", default=[])
    parser.add_argument("--qlever-endpoint", type=str, default=None)
    sample_group = parser.add_argument_group("samples")
    sample_group.add_argument(
        "--max-questions",
        type=int,
        default=None
    )
    sample_group.add_argument(
        "--samples-per-question",
        type=int,
        default=1
    )
    sample_group.add_argument(
        "--sample-dropout",
        type=float,
        default=0.1
    )
    sample_group.add_argument(
        "--selections-min-k",
        type=int,
        default=8
    )
    sample_group.add_argument(
        "--selections-max-k",
        type=int,
        default=32
    )
    sample_group.add_argument(
        "--selections-min-aliases",
        type=int,
        default=0
    )
    sample_group.add_argument(
        "--selections-max-aliases",
        type=int,
        default=8
    )
    sample_group.add_argument(
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
        data = load_dataset(
            "third_party/KGQA-datasets/lcquad_v2",
            "lcquad2-wikidata"
        )
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

    elif args.grail_qa:
        kg = "freebase"
        data = load_dataset(
            "third_party/KGQA-datasets/grail_qa",
            "grail_qa"
        )

        output["train"] = [
            Sample(item["question"], item["sparql_query"])
            for item in data["train"]
        ]
        output["val"] = [
            Sample(item["question"], item["sparql_query"])
            for item in data["validation"]
        ]

        data = load_dataset(
            "third_party/KGQA-datasets/grail_qa",
            "grailqa_test_public"
        )
        output["test"] = [
            Sample(item["question"], "")
            for item in data["test"]
        ]

    elif args.wqsp:
        data = load_dataset("third_party/KGQA-datasets/webqsp")
        kg = "freebase"
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                for sparql in item["Parses"]["Sparql"]:
                    samples.append(Sample(
                        item["RawQuestion"],
                        sparql
                    ))
            output[split] = samples

    elif args.cwq:
        data = load_dataset(
            "third_party/KGQA-datasets/complex_web_questions",
            "complex_web_questions"
        )
        kg = "freebase"
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                samples.append(Sample(
                    item["question"],
                    item["sparql"]
                ))
            output[split] = samples

        data = load_dataset(
            "third_party/KGQA-datasets/complex_web_questions",
            "complexwebquestions_test"
        )
        output["test"] = [
            Sample(item["question"], item["sparql"])
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
                samples.append(Sample(
                    item["question"],
                    item["sparql"].replace(" ns:", " fb:")
                ))
            output[s] = samples

    elif args.lc_quad1_dbpedia:
        kg = "dbpedia"
        data = load_dataset(
            "third_party/KGQA-datasets/lcquad_v1",
            "lcquad"
        )
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                samples.append(Sample(
                    item["corrected_question"].replace(" ?", "?"),
                    item["sparql_query"]
                ))
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

            output[split] = samples

    elif args.dblp_quad:
        kg = "dblp"
        data = load_dataset("awalesushil/DBLP-QuAD")
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                if split != "train" and not item["held_out"]:
                    continue
                samples.append(Sample(
                    item["question"]["string"],
                    item["query"]["sparql"]
                ))
                samples.append(Sample(
                    item["paraphrased_question"]["string"],
                    item["query"]["sparql"]
                ))
            output[split] = samples

    else:
        raise RuntimeError("unknown dataset")

    return kg, output


# nltk english stopwords
STOP = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
    'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
]


def get_search_query(name: str, index_type: str) -> str:
    keywords = normalize(name).split()
    if index_type == "prefix":
        keywords = list(
            k for k in set(keywords)
            if k not in STOP
            and not any(
                k.startswith(other) and k != other
                for other in keywords
            )
        )
        # limit to at most 5 keywords
        keywords = random.sample(keywords, min(5, len(keywords)))

    elif index_type == "qgram" and len(keywords) > 3:
        start = random.randint(0, len(keywords) - 3)
        end = random.randint(start + 3, len(keywords))
        keywords = keywords[start:end]

    return " ".join(keywords)


def prepare_stages(
    question: str,
    sparql: str,
    managers: list[KgManager],
    args: argparse.Namespace
) -> list[tuple[str | Chat, str]]:
    manager = random.choice(managers)
    sparql_encoded = sparql.encode()
    parse = manager.parser.parse(
        sparql,
        skip_empty=True,
        collapse_single=False
    )

    mappings = {
        "entity": manager.entity_mapping,
        "property": manager.property_mapping
    }
    indices = {
        "entity": manager.entity_index,
        "property": manager.property_index
    }

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
            return label, None, label, [], None

        if obj["name"] == "BooleanLiteral":
            label = obj["children"][0]["children"][0]
            return label, None, label, [], None

        if obj["name"] != "iri":
            return None

        child = obj["children"][0]
        if child["name"] != "PrefixedName":
            return None

        child = child["children"][0]
        if child["name"] != "PNAME_LN":
            return None

        pfx, val = child["value"].split(":", 1)
        if pfx in manager.prefixes:
            return child["value"], None, child["value"], [], None

        elif pfx in manager.custom_prefixes:
            # check whether the iri is a valid entity or property
            long = manager.custom_prefixes[pfx]
            iri = long + val + ">"

            matching = {}
            for obj_type in ["entity", "property"]:
                map = mappings[obj_type]
                index = indices[obj_type]
                norm = map.normalize(iri)
                if norm is None or norm[0] not in map:
                    continue

                id = map[norm[0]]
                label = index.get_name(id)
                syns = [
                    s for s in index.get_val(id, 2).split(";;;")
                    if s != ""
                ]
                matching[obj_type] = (*norm, label, syns)

            if not matching:
                return None

            nxt = next(val for val in matching.values())
            return *nxt, {"matching": list(matching)}

    items = [
        (item, map_item(item))
        for item in find_all(
            parse,
            name={"iri", "RDFLiteral", "NumericLiteral", "BooleanLiteral"},
            skip={"Prologue"}
        )
    ]
    # filter out invalid items
    items = [
        (item, processed)
        for item, processed in items
        if processed is not None
    ]

    samples = []
    for item_idx in random.sample(
        list(range(len(items))),
        min(len(items), args.samples_per_question)
    ):
        manager = random.choice(managers)
        item, processed = items[item_idx]

        start, end = span(item)
        assert end >= start, "invalid span"
        identifier, variant, label, syns, info = processed

        prefix_raw = sparql_encoded[:start].decode()
        prefix, _ = manager.replace_iris(
            prefix_raw,
            is_prefix=True
        )

        if item_idx > 0:
            last, _ = items[item_idx - 1]
            (_, last_end) = span(last)
            last_prefix_raw = sparql_encoded[:last_end].decode()
            last_prefix, _ = manager.replace_iris(
                last_prefix_raw,
                is_prefix=True
            )
        else:
            last_prefix = ""

        if item_idx == len(items) - 1:
            # add additional continuation sample for the end of the query
            final_prefix, _ = manager.replace_iris(
                sparql_encoded[:end].decode(),
                is_prefix=True
            )
            final_continuation = sparql_encoded[end:].decode()
            final_continuation_prompt = manager.get_sparql_continuation_prompt(
                question,
                final_prefix
            )
            samples.append((final_continuation_prompt, final_continuation))

        continuation_prompt = manager.get_sparql_continuation_prompt(
            question,
            last_prefix
        )
        continuation = prefix[len(last_prefix):] + SEARCH_TOKEN
        samples.append((continuation_prompt, continuation))

        random.shuffle(syns)
        all_syns = [label] + syns

        search_failures = set()
        num_search_failures = random.sample(
            list(range(len(all_syns) + 1)),
            1,
            counts=list(range(len(all_syns) + 1, 0, -1))
        )[0]
        for i in range(num_search_failures):
            search_failures.add(get_search_query(
                all_syns[i],
                manager.entity_index.get_type()
            ))

        search = all_syns[min(num_search_failures, len(all_syns) - 1)]

        search_prompt, _ = manager.get_search_prompt_and_regex(
            question,
            prefix,
            failures=search_failures
        )

        samples.append((search_prompt, search))

        selection_k = random.randint(
            args.selections_min_k,
            args.selections_max_k
        )

        alts = manager.get_selection_alternatives(
            prefix_raw,
            search,
            selection_k,
            max_candidates=16384,
            endpoint=args.qlever_endpoint,
            max_retries=3
        )
        if alts is None:
            alts = {}

        target_alts = [
            (obj_type, alt)
            for obj_type, obj_alts in alts.items()
            for alt in obj_alts
            if alt.identifier == identifier
            and (variant is None or variant in alt.variants)
        ]
        target_alt = random.choice(target_alts) if target_alts else None

        select_failures = set()

        drop_alt = random.random() < args.sample_dropout
        if target_alt is not None and drop_alt:
            # differentiate between dropping the target from
            # the list of alternatives entirely or only adding the
            # variant to previous fails
            target_obj_type, target_alternative = target_alt
            if random.random() < 0.5:
                alts[target_obj_type].remove(target_alternative)
            else:
                select_failures.add((target_obj_type, identifier, variant))

            target_alt = None

        alts_to_fail = [
            (obj_type, alt.identifier, var)
            for obj_type, obj_alts in alts.items()
            for alt in obj_alts
            for var in (alt.variants or [None])
            if (target_alt is None
                or alt.identifier != target_alt[1].identifier
                or var != variant)
            and (target_alt is None or obj_type == target_alt[0])
        ]

        num_select_failures = random.sample(
            list(range(3 - len(select_failures) + 1)),
            1,
            counts=list(range(3 - len(select_failures) + 1, 0, -1))
        )[0]
        failed = random.sample(
            alts_to_fail,
            min(len(alts_to_fail), num_select_failures),
        )
        select_failures.update(set(failed))

        select_prompt, _ = manager.get_selection_prompt_and_regex(
            question,
            prefix,
            search,
            alts,
            max_aliases=random.randint(
                args.selections_min_aliases,
                args.selections_max_aliases
            ),
            add_infos=random.random() < args.selections_add_info_p,
            failures=select_failures
        )

        num_alts = sum(len(obj_alts) for obj_alts in alts.values())
        if target_alt is None:
            selection = f"{num_alts}. none"
        else:
            offset = sum(
                len(alts[obj_type])
                for obj_type in OBJ_TYPES[:OBJ_TYPES.index(target_alt[0])]
            )
            select_idx = offset + alts[target_alt[0]].index(target_alt[1])
            select_name = target_alt[1].label
            if variant:
                select_name += f" ({variant})"
            selection = f"{select_idx + 1}. {select_name}"

        samples.append((select_prompt, selection))

    return samples


def prepare_sample(
    sample: Sample,
    args: argparse.Namespace,
    split: str
) -> tuple[str, str, list[tuple[str | Chat, str]]] | None:
    global managers

    # clean sparql in sample
    sample = Sample(
        clean(sample.question),
        clean(sample.sparql),
    )

    manager = random.choice(managers)

    try:
        raw_sparql = manager.fix_prefixes(sample.sparql)
    except Exception:
        return None

    if split == "test":
        return sample.question, raw_sparql, []

    prompt = manager.get_sparql_prompt(sample.question)
    sparql, inc = manager.replace_iris(
        raw_sparql,
        with_iri=False
    )
    if inc:
        return None

    sparqls: list[tuple[str | Chat, str]] = [(prompt, sparql)]
    sparqls.extend(prepare_stages(
        sample.question,
        raw_sparql,
        managers,
        args
    ))

    return sample.question, raw_sparql, sparqls


def prepare_sample_mp(args: tuple[Sample, argparse.Namespace, str]):
    return prepare_sample(*args)


managers: list[KgManager] = []


def init(kg: str, args: argparse.Namespace):
    global managers
    random.seed(args.seed)

    ent_indices = []
    prop_indices = []
    for index_type in ["qgram", "prefix"]:
        ent_index, ent_mapping = load_index_and_mapping(
            args.entities,
            index_type
        )
        prop_index, prop_mapping = load_index_and_mapping(
            args.properties,
            index_type,
            WikidataPropertyMapping if kg == "wikidata" else None
        )
        ent_indices.append((ent_index, ent_mapping))
        prop_indices.append((prop_index, prop_mapping))

    for ent, prop in zip(ent_indices, prop_indices):
        ent_index, ent_mapping = ent
        prop_index, prop_mapping = prop
        manager = get_kg_manager(
            kg,
            ent_index,
            prop_index,
            ent_mapping,
            prop_mapping,
        )
        managers.append(manager)


def prepare(args: argparse.Namespace):
    logging.basicConfig(
        format="[%(asctime)s] {%(name)s - %(levelname)s} %(message)s",
        level=logging.INFO
    )
    kg, data = load_data(args)
    num_samples = {s: len(samples) for s, samples in data.items()}
    print(f"Number of raw samples: {num_samples}")

    os.makedirs(args.output, exist_ok=True)

    if args.num_workers is None:
        args.num_workers = min(mp.cpu_count(), 8)

    print(f"Starting {args.num_workers} workers")
    pool = mp.Pool(
        args.num_workers,
        initializer=init,
        initargs=(kg, args)
    )

    for split, samples in data.items():
        if split in args.skip:
            print(f"skipping {split} split")
            continue

        if args.max_questions is not None:
            samples = random.sample(
                samples,
                min(len(samples), args.max_questions)
            )

        input = os.path.join(
            args.output,
            f"{split}_input.jsonl"
        )
        target = os.path.join(
            args.output,
            f"{split}_target.jsonl"
        )
        raw = os.path.join(
            args.output,
            f"{split}_raw.jsonl"
        )
        invalid = 0
        num_sparqls = 0
        with open(input, "w") as inf, \
                open(target, "w") as tf, \
                open(raw, "w") as rf:
            for output in tqdm(
                pool.imap(
                    prepare_sample_mp,
                    ((sample, args, split) for sample in samples)
                ),  # type: ignore
                desc=f"processing and writing {split} samples",
                leave=False,
                total=len(samples),
                disable=not args.progress
            ):
                if output is None:
                    invalid += 1
                    continue

                (question, raw_sparql, sparqls) = output
                rf.write(json.dumps({
                    "question": question,
                    "sparql": raw_sparql
                }) + "\n")

                if split == "test":
                    assert len(sparqls) == 0
                    inf.write(json.dumps(question) + "\n")
                    tf.write(json.dumps(raw_sparql) + "\n")
                    continue

                num_sparqls += len(sparqls)
                for prompt, target in sparqls:
                    if isinstance(prompt, str):
                        prompt = [{
                            "role": "user",
                            "text": prompt
                        }]
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
