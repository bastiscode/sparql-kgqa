import argparse
import os
import json

from tqdm import tqdm
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OpenHermes dataset")
    parser.add_argument("output", type=str, help="Output dir")
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> None:
    input_file = os.path.join(args.output, "train_input.jsonl")
    target_file = os.path.join(args.output, "train_target.jsonl")
    if os.path.exists(input_file) and os.path.exists(target_file):
        print("already exists")
        return

    dataset = load_dataset("teknium/OpenHermes-2.5", split="train")

    with open(input_file, "w") as input_f, open(target_file, "w") as target_f:
        for sample in tqdm(dataset, "preparing OpenHermes 2.5", leave=False):
            assert isinstance(sample, dict)
            if len(sample["conversations"]) < 2:
                continue

            chat = []
            skip = False
            for msg in sample["conversations"][:-1]:
                if msg["from"] == "human":
                    chat.append({
                        "role": "user",
                        "text": msg["value"]
                    })
                elif msg["from"] == "gpt":
                    chat.append({
                        "role": "assistant",
                        "text": msg["value"]
                    })
                elif msg["from"] == "system":
                    chat.append({
                        "role": "system",
                        "text": msg["value"]
                    })
                else:
                    skip = True
                    break

            if skip:
                continue

            last = sample["conversations"][-1]
            if last["from"] != "gpt":
                continue

            input_f.write(json.dumps(chat) + "\n")
            target_f.write(json.dumps(last["value"]) + "\n")


if __name__ == "__main__":
    prepare(parse_args())
