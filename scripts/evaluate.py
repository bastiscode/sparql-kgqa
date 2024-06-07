import argparse
import os

from tqdm import tqdm

from text_utils.io import load_text_file


from sparql_kgqa.sparql.utils import (
    QLEVER_URLS,
    calc_f1,
    load_sparql_parser
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--prediction", type=str, required=True)
    parser.add_argument("--save-invalid", type=str, default=None)
    parser.add_argument("--save-incorrect", type=str, default=None)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--empty-target-invalid", action="store_true")
    parser.add_argument(
        "--kg",
        type=str,
        choices=list(QLEVER_URLS),
        default="wikidata"
    )
    parser.add_argument("--qlever-endpoint", type=str, default=None)
    return parser.parse_args()


def delete_file_or_create_dir(path: str):
    if os.path.exists(path):
        os.remove(path)
    else:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)


def evaluate(args: argparse.Namespace):
    targets = load_text_file(args.target)
    predictions = load_text_file(args.prediction)
    if not args.allow_subset:
        assert len(targets) == len(predictions), \
            "expected same number of predictions and targets"

    if args.save_invalid or args.save_incorrect:
        inputs = load_text_file(args.input)
        assert len(inputs) == len(targets), \
            "expected same number of inputs and targets"
    else:
        inputs = []

    if args.save_invalid:
        delete_file_or_create_dir(args.save_invalid)

    if args.save_incorrect:
        delete_file_or_create_dir(args.save_incorrect)

    parser = load_sparql_parser([args.kg])
    f1s = []
    pred_invalid = 0
    tgt_invalid = 0
    for i, (pred, target) in tqdm(
        enumerate(zip(predictions, targets)),
        desc="evaluating",
        total=len(predictions),
        leave=False
    ):
        f1, pred_inv, tgt_inv = calc_f1(
            pred,
            target,
            parser,
            allow_empty_target=not args.empty_target_invalid,
            kg=args.kg,
            qlever_endpoint=args.qlever_endpoint
        )
        if args.save_invalid and f1 is None:
            with open(args.save_invalid, "a", encoding="utf8") as f:
                f.write(
                    f"{i+1}.\n"
                    f"input : {inputs[i]}\n"
                    f"pred  : {predictions[i]}\n"
                    f"target: {targets[i]}\n\n"
                )
        if args.save_incorrect and f1 is not None and f1 < 1.0:
            with open(args.save_incorrect, "a", encoding="utf8") as f:
                f.write(
                    f"{i+1}.\n"
                    f"input : {inputs[i]}\n"
                    f"pred  : {predictions[i]}\n"
                    f"target: {targets[i]}\n\n"
                )

        if pred_inv:
            pred_invalid += 1
            f1 = 0.0
        if tgt_inv:
            tgt_invalid += 1
            f1 = 0.0
        f1s.append(f1)

    print(
        f"Query-averaged F1: {sum(f1s) / len(f1s):.2%} "
        f"({pred_invalid:,} invalid predictions, "
        f"{pred_invalid / len(f1s):.2%} | "
        f"{tgt_invalid:,} invalid targets, "
        f"{tgt_invalid / len(f1s):.2%})"
    )


if __name__ == "__main__":
    evaluate(parse_args())
