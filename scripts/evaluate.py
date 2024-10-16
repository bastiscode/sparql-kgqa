import argparse
import os
import json
from typing import Any
from collections import Counter

from tqdm import tqdm

from text_utils.io import load_text_file


from sparql_kgqa.sparql.utils2 import (
    QLEVER_URLS,
    AskResult,
    KgManager,
    run_parallel,
    get_index_dir,
    load_index_and_mapping,
    get_kg_manager,
    WikidataPropertyMapping
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--prediction", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--empty-target-invalid", action="store_true")
    parser.add_argument(
        "-kg",
        "--knowledge_graph",
        type=str,
        choices=list(QLEVER_URLS),
        default="wikidata"
    )
    parser.add_argument("--entities", type=str)
    parser.add_argument("--properties", type=str)
    parser.add_argument("--index-type", type=str,
                        choices=["prefix", "qgram"],
                        default="prefix")
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("-n", "--num-workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--prediction-format",
        type=str,
        choices=["text", "jsonl"],
        default="jsonl"
    )
    return parser.parse_args()


def create_dir(path: str):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def get_result_set_or_error(
    sparql: str,
    manager: KgManager,
    endpoint: str | None = None,
    timeout: float | None = None,
    max_retries: int = 1
) -> tuple[Counter | None, str | None]:
    try:
        result = manager.execute_sparql(
            sparql,
            endpoint,
            timeout,
            max_retries
        )
    except Exception as e:
        return None, str(e)

    if isinstance(result, AskResult):
        return Counter({result: 1}), None
    else:
        return Counter(
            tuple(result[i])
            for i in range(1, len(result))
        ), None


def calc_f1(
    pred: str,
    target: str,
    manager: KgManager,
    allow_empty_target: bool = True,
    endpoint: str | None = None,
    timeout: float | None = None,
    max_retries: int = 1
) -> tuple[float | None, str | Counter, str | Counter]:
    pred_set, pred_err = get_result_set_or_error(
        pred,
        manager,
        endpoint,
        timeout,
        max_retries
    )
    target_set, target_err = get_result_set_or_error(
        target,
        manager,
        endpoint,
        timeout,
        max_retries
    )
    if pred_err is not None or target_err is not None:
        return (
            None,
            pred_err or pred_set,
            target_err or target_set
        )  # type: ignore

    assert pred_set is not None and target_set is not None
    if len(target_set) == 0 and not allow_empty_target:
        return None, pred_set, "target set is empty"

    if len(pred_set) == 0 and len(target_set) == 0:
        return 1.0, pred_set, target_set

    tp = (pred_set & target_set).total()
    fp = (pred_set - target_set).total()
    fn = (target_set - pred_set).total()
    # calculate precision, recall and f1
    if tp > 0:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0.0

    return f1, pred_set, target_set


def evaluate(args: argparse.Namespace):
    base = os.path.splitext(args.prediction)[0]
    result_file = f"{base}.result.json"

    if os.path.exists(result_file) and not args.overwrite:
        with open(result_file, "r") as inf:
            result = json.load(inf)

        mean_f1 = result["scores"]["f1"]["mean"]
        f1s = result["scores"]["f1"]["values"]
        mean_em = result["scores"]["em"]["mean"]
        ems = result["scores"]["em"]["values"]
        invalid = result["invalid"]
        incorrect = result["incorrect"]
        num_samples = result["num_samples"]

    else:
        targets = load_text_file(args.target)
        targets = [json.loads(t) for t in targets]
        inputs = load_text_file(args.input)
        inputs = [json.loads(i) for i in inputs]
        assert len(inputs) == len(targets), \
            "expected same number of inputs and targets"

        predictions = load_text_file(args.prediction)
        if args.prediction_format == "jsonl":
            predictions = [json.loads(p) for p in predictions]

        if not args.allow_subset:
            assert len(targets) == len(predictions), \
                "expected same number of predictions and targets"

        if args.limit:
            targets = targets[:args.limit]
            inputs = inputs[:args.limit]
            predictions = predictions[:args.limit]

        kg = args.knowledge_graph
        index_dir = get_index_dir()
        if args.entities is None:
            assert index_dir is not None, \
                "SEARCH_INDEX_DIR environment variable must be set if " \
                "--entities is not provided"
            args.entities = os.path.join(index_dir, f"{kg}-entities")

        if args.properties is None:
            assert index_dir is not None, \
                "SEARCH_INDEX_DIR environment variable must be set if " \
                "--properties is not provided"
            args.properties = os.path.join(index_dir, f"{kg}-properties")

        ent_index, ent_mapping = load_index_and_mapping(
            args.entities,
            args.index_type,
        )
        is_wikidata = args.knowledge_graph == "wikidata"
        prop_index, prop_mapping = load_index_and_mapping(
            args.properties,
            args.index_type,
            # wikidata properties need special mapping
            # because of wdt, p, ps, pq, ... variants
            WikidataPropertyMapping if is_wikidata else None,
        )

        manager = get_kg_manager(
            kg,
            ent_index,
            prop_index,
            ent_mapping,
            prop_mapping
        )

        incorrect = {}
        invalid = {}
        f1s = []
        iter = (
            (pred, target, manager,
             not args.empty_target_invalid,
             args.endpoint, args.timeout, args.max_retries)
            for pred, target in zip(predictions, targets)
        )
        for i, (f1, pred, tgt) in tqdm(
            enumerate(run_parallel(calc_f1, iter, args.num_workers)),
            desc="evaluating",
            total=len(predictions),
            leave=False
        ):
            invalid_pred = isinstance(pred, str)
            if invalid_pred:
                invalid[i] = {"prediction": pred}
                f1 = 0.0

            if isinstance(tgt, str):
                if i not in invalid:
                    invalid[i] = {}
                invalid[i]["target"] = tgt
                continue

            assert f1 is not None

            if f1 < 1.0 and not invalid_pred:
                incorrect[i] = {"f1": f1}

            f1s.append(f1)

        mean_f1 = sum(f1s) / max(len(f1s), 1)
        base = os.path.splitext(args.prediction)[0]
        result_file = f"{base}.result.json"
        create_dir(result_file)

        def format_indices_and_infos(
            indices: dict[int, dict[str, Any]]
        ) -> list[dict]:
            return [
                {
                    "sample": i + 1,
                    "input": inputs[i],
                    "target": targets[i],
                    "prediction": predictions[i],
                    "infos": infos
                }
                for i, infos in indices.items()
            ]

        ems = [int(f1 == 1.0) for f1 in f1s]
        mean_em = sum(ems) / max(1, len(ems))

        invalid = format_indices_and_infos(invalid)
        incorrect = format_indices_and_infos(incorrect)
        num_samples = len(predictions)
        with open(result_file, "w") as outf:
            json.dump(
                {
                    "num_samples": num_samples,
                    "scores": {
                        "f1": {
                            "mean": mean_f1,
                            "values": f1s
                        },
                        "em": {
                            "mean": mean_em,
                            "values": ems
                        }
                    },
                    "invalid": invalid,
                    "incorrect": incorrect
                },
                outf,
                indent=2
            )

    pred_invalid = sum(
        int("prediction" in item["infos"])
        for item in invalid
    )
    tgt_invalid = sum(
        int("target" in item["infos"])
        for item in invalid
    )
    print(
        f"Mean F1: {mean_f1:.2%}\n"
        f"Mean EM: {mean_em:.2%}\n"
        f"({pred_invalid:,}/{num_samples:,} pred error, "
        f"{pred_invalid/max(1, num_samples):.2%} | "
        f"{tgt_invalid:,}/{num_samples:,} tgt error, "
        f"{tgt_invalid/max(1, num_samples):.2%} | "
        f"{len(incorrect):,}/{len(f1s):,} pred incorrect, "
        f"{len(incorrect)/max(1, len(f1s)):.2%} )"
    )


if __name__ == "__main__":
    evaluate(parse_args())
