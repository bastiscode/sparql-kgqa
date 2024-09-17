import argparse
import random
import os
import re
import json
import collections
from typing import Any

from search_index import PrefixIndex, QGramIndex, normalize
from search_index.index import SearchIndex
from tqdm import tqdm
from datasets import load_dataset

from sparql_kgqa.sparql.utils import find_all
from sparql_kgqa.sparql.utils2 import (
    Chat,
    KgManager,
    WikidataPropertyMapping,
    get_kg_manager,
    clean,
    format_obj_type,
    load_index_and_mapping,
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
    parser.add_argument("--properties", type=str, required=True)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip", nargs="+", default=[])
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
        default=4
    )
    sample_group.add_argument(
        "--selections-max-k",
        type=int,
        default=16
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
        # todo: load this with kqa datasets
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

    elif args.grail_qa is not None:
        kg = "freebase"
        samples = []
        data = load_dataset(args.grail_qa, "grail_qa")

        output["train"] = [
            Sample(item["question"], item["sparql_query"])
            for item in data["train"]
        ]
        output["val"] = [
            Sample(item["question"], item["sparql_query"])
            for item in data["validation"]
        ]

        data = load_dataset(args.grail_qa, "grailqa_test_public")
        output["test"] = [
            Sample(item["question"], "")
            for item in data["test"]
        ]

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


def filter_stopwords(words: list[str]) -> list[str]:
    return [
        word for word in words
        if word not in STOP
    ]


def get_search_query(
    id: int,
    index: SearchIndex,
    k: int,
    return_non_matching_ids: bool = False
) -> tuple[str, list[int]]:
    data = index.get_row(id)
    name, _, syns, *_ = data.split("\t")
    syns = [s for s in syns.split(";;;") if s != ""]

    # simulate some sensible search behavior
    # for the different index types
    if isinstance(index, PrefixIndex):
        keywords = set(normalize(name).split())
        n_syns = random.randint(0, min(1, len(syns)))
        for syn in random.sample(syns, n_syns):
            keywords.update(normalize(syn).split())
        keywords = list(
            k for k in keywords
            if k not in STOP
            and not any(
                k.startswith(other) for other in keywords
                if k != other
            )
        )
        random.shuffle(keywords)
        # limit to at most 5 keywords
        query = " ".join(keywords[:5])
    else:
        assert isinstance(index, QGramIndex)
        syns.append(name)
        # make label equally likely to be selected
        # as all other synonyms
        if len(syns) > 1:
            counts = [1] * len(syns)
            counts[-1] = len(syns) - 1
        else:
            counts = [1]
        query = normalize(random.sample(syns, 1, counts=counts)[0])

    non_matching_ids = []
    if not return_non_matching_ids:
        return query, non_matching_ids

    matches = index.find_matches(query)
    for match_id, _ in matches[:k]:
        if id == match_id:
            continue

        non_matching_ids.append(match_id)

    return query, non_matching_ids


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

    def map_item(obj) -> tuple[Any, str, str, tuple[int, int]] | None:
        if obj["name"] != "iri":
            return None
        child = obj["children"][0]
        if child["name"] != "PrefixedName":
            return None
        child = child["children"][0]
        if child["name"] != "PNAME_LN":
            return None
        pfx, val = child["value"].split(":", 1)
        return obj, pfx, val, child["byte_span"]

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

    samples = []
    for item_idx in random.sample(
        list(range(len(items))),
        min(len(items), args.samples_per_question)
    ):
        manager = random.choice(managers)
        item = items[item_idx]
        assert item is not None

        item, pfx, iri, (start, end) = item
        long = manager.custom_prefixes[pfx]
        iri = long + iri + ">"

        index_map = manager.entity_mapping
        index = manager.entity_index
        norm = index_map.normalize(iri)
        obj_type = "entity"
        if norm is None or norm[0] not in index_map:
            index_map = manager.property_mapping
            index = manager.property_index
            norm = index_map.normalize(iri)
            obj_type = "property"

        search_token = format_obj_type(obj_type)

        if norm is None or norm[0] not in index_map:
            continue

        key, variant = norm

        prefix_raw = sparql_encoded[:start].decode(errors="replace")
        prefix, _ = manager.replace_iris(
            prefix_raw,
            is_prefix=True
        )

        if item_idx > 0:
            last = items[item_idx - 1]
            assert last is not None
            *_, (_, last_end) = last
            last_prefix_raw = sparql_encoded[:last_end].decode(
                errors="replace")
            last_prefix, _ = manager.replace_iris(
                last_prefix_raw,
                is_prefix=True
            )
        else:
            last_prefix = ""

        if item_idx == len(items) - 1:
            # add additional continuation sample for the end of the query
            final_prefix, _ = manager.replace_iris(
                sparql_encoded[:end].decode(errors="replace"),
                is_prefix=True
            )
            final_continuation = sparql_encoded[end:].decode(errors="ignore")
            final_continuation_prompt = manager.get_sparql_continuation_prompt(
                question,
                final_prefix
            )
            samples.append(
                (final_continuation_prompt, final_continuation)
            )

        continuation_prompt = manager.get_sparql_continuation_prompt(
            question,
            last_prefix
        )
        continuation = prefix[len(last_prefix):] + search_token
        samples.append((continuation_prompt, continuation))

        search_failures = set()
        selection_k = random.randint(
            args.selections_min_k,
            args.selections_max_k
        )
        search, non_matching_ids = get_search_query(
            index_map[key],
            index,
            selection_k,
            return_non_matching_ids=True
        )
        num_search_failures = min(
            random.randint(0, 3),
            len(non_matching_ids)
        )
        for id in random.sample(non_matching_ids, num_search_failures):
            search_fail, *_ = get_search_query(
                id,
                index,
                selection_k
            )
            search_failures.add(search_fail)

        drop_search = random.random() < args.sample_dropout
        if drop_search:
            search = normalize(index.get_name(index_map[key]))
            search_failures.add(search)

        search_prompt, _ = manager.get_search_prompt_and_regex(
            question,
            obj_type,
            prefix,
            failures=search_failures
        )

        samples.append((search_prompt, search))

        alts = manager.get_selection_alternatives(
            prefix_raw,
            obj_type,
            search,
            selection_k,
            max_candidates=16384
        )
        if alts is None:
            alts = []

        target_alt = next(
            (alt for alt in alts
             if alt.identifier == key
             and variant in (alt.variants or [None])),
            None
        )

        select_failures = set()

        drop_alt = random.random() < args.sample_dropout
        if target_alt is not None and drop_alt:
            # differentiate between dropping the target from
            # the list of alternatives entirely or only adding the
            # variant to previous fails
            if random.random() < 0.5:
                alts.remove(target_alt)
            else:
                select_failures.add(iri)

            target_alt = None

        other_alts = [
            (alt.identifier, var)
            for alt in alts
            for var in (alt.variants or [None])
            if target_alt is None
            or alt.identifier != target_alt.identifier
            or var != variant
        ]

        if other_alts:
            counts = [
                max(1, int(index.get_val(index_map[id], 1)))
                for id, _ in other_alts
            ]
            num_select_failures = random.randint(0, 3 - len(select_failures))
            failed = random.sample(
                other_alts,
                min(len(other_alts), num_select_failures),
                counts=counts
            )
            select_failures.update(
                index_map.denormalize(*fail)
                for fail in failed
            )

        select_prompt, _ = manager.get_selection_prompt_and_regex(
            question,
            prefix,
            obj_type,
            search,
            alts,
            max_aliases=random.randint(
                args.selections_min_aliases,
                args.selections_max_aliases
            ),
            add_infos=random.random() < args.selections_add_info_p,
            failures=select_failures
        )

        if target_alt is None:
            selection = f"{len(alts)}. none"
        else:
            select_idx = alts.index(target_alt)
            select_name = target_alt.label
            if variant:
                select_name += f" ({variant})"
            selection = f"{select_idx + 1}. {select_name}"

        samples.append((select_prompt, selection))

    return samples


def prepare_sample(
    sample: Sample,
    managers: list[KgManager],
    args: argparse.Namespace,
    split: str
) -> tuple[str, str, list[tuple[str | Chat, str]]] | None:
    # clean sparql in sample
    sample = Sample(
        sample.question,
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
    sparql, inc = manager.replace_iris(raw_sparql)
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


def prepare(args: argparse.Namespace):
    random.seed(args.seed)
    kg, data = load_data(args)

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

    managers = []
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

    os.makedirs(args.output, exist_ok=True)

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
                # run_parallel(
                #     prepare_sample,
                #     ((sample, managers, args, split) for sample in samples),
                #     args.num_workers,
                # ),
                (prepare_sample(sample, managers, args, split)
                 for sample in samples),
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
