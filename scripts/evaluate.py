import argparse
import os
import json

from tqdm import tqdm

from text_utils.io import load_text_file
from text_utils import grammar


from sparql_kgqa.sparql.utils import (
    QLEVER_URLS,
    calc_f1
)
from sparql_kgqa.sparql.utils2 import (
    run_parallel,
    load_sparql_grammar,
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
        "--kg",
        type=str,
        choices=list(QLEVER_URLS),
        default="wikidata"
    )
    parser.add_argument("--qlever-endpoint", type=str, default=None)
    parser.add_argument("-n", "--num-workers", type=int, default=None)
    parser.add_argument(
        "--prediction-format",
        type=str,
        choices=["text", "jsonl"],
        default="text"
    )
    return parser.parse_args()


def create_dir(path: str):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def evaluate(args: argparse.Namespace):
    base = os.path.splitext(args.prediction)[0]
    result_file = f"{base}.result.json"
    if os.path.exists(result_file) and not args.overwrite:
        with open(result_file, "r") as inf:
            result = json.load(inf)

        mean_f1 = result["scores"]["f1"]["mean"]
        f1s = result["scores"]["f1"]["values"]
        pred_invalid = result["invalid_predictions"]
        tgt_invalid = result["invalid_targets"]
        incorrect = result["incorrect_predictions"]

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

        gram, lex = load_sparql_grammar()
        parser = grammar.LR1Parser(gram, lex)

        incorrect = []
        pred_invalid = []
        tgt_invalid = []
        f1s = []
        iter = (
            (pred, target, parser,
             not args.empty_target_invalid, args.kg, args.qlever_endpoint)
            for pred, target in zip(predictions, targets)
        )
        for i, (f1, pred_inv, tgt_inv) in tqdm(
            enumerate(run_parallel(calc_f1, iter, args.num_workers)),
            desc="evaluating",
            total=len(predictions),
            leave=False
        ):
            if pred_inv:
                pred_invalid.append(i)
                f1 = 0.0
            if tgt_inv:
                tgt_invalid.append(i)
                continue
            if f1 < 1.0:
                incorrect.append(i)
            f1s.append(f1)

        mean_f1 = sum(f1s) / len(f1s)
        base = os.path.splitext(args.prediction)[0]
        result_file = f"{base}.result.json"
        create_dir(result_file)

        def format_indices(indices: list[int]) -> list[dict]:
            return [
                {
                    "sample": i + 1,
                    "input": inputs[i],
                    "target": targets[i],
                    "prediction": predictions[i]
                }
                for i in indices
            ]

        with open(result_file, "w") as outf:
            json.dump(
                {
                    "scores": {
                        "f1": {
                            "mean": mean_f1,
                            "values": f1s
                        }
                    },
                    "invalid_predictions": format_indices(pred_invalid),
                    "invalid_targets": format_indices(tgt_invalid),
                    "incorrect_predictions": format_indices(incorrect),
                },
                outf,
                indent=2
            )

    print(
        f"Query-averaged F1: {mean_f1:.2%} "
        f"({len(pred_invalid):,} invalid predictions, "
        f"{len(pred_invalid) / len(f1s):.2%} | "
        f"{len(tgt_invalid):,} invalid targets, "
        f"{len(tgt_invalid) / len(f1s):.2%})"
    )


if __name__ == "__main__":
    evaluate(parse_args())
