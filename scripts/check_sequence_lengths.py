import argparse
import json
from tqdm import tqdm

import numpy as np


from text_utils import configuration, tokenization

from sparql_kgqa.api.utils import format_chat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("experiment", type=str)
    parser.add_argument("length", type=int)
    return parser.parse_args()


def check_lengths(args: argparse.Namespace):
    with open(args.input, "r") as inf:
        input_data = [json.loads(s.strip()) for s in inf]

    with open(args.target, "r") as inf:
        target_data = [json.loads(s.strip()) for s in inf]

    cfg = configuration.load_config_from_experiment(args.experiment)
    tokenizer = tokenization.Tokenizer.from_config(
        cfg["inference"]["tokenizer"]
    )
    chat_template = cfg["inference"]["chat_template"]

    lengths = []
    chats = []
    for chat, cont in tqdm(
        zip(input_data, target_data),
        desc="checking sequences",
        leave=False,
        total=len(input_data)
    ):
        if chat[-1].get("partial", False):
            chat[-1]["text"] += cont
        else:
            chat.append({"role": "assistant", "text": cont})

        chat = format_chat(chat, chat_template)
        token_ids = tokenizer.tokenize(chat, True).token_ids
        lengths.append(len(token_ids))
        chats.append(chat)

    chats_and_lengths = sorted(zip(lengths, chats), key=lambda item: item[0])
    for chat, length in chats_and_lengths[-3:]:
        print(chat)
        print(length)
        print("-" * 40)

    lengths = np.array(lengths)
    print(f"Avg length: {np.mean(lengths):.2f}")
    print(f"Med length: {np.median(lengths)}")
    print(f"95th percentile: {np.percentile(lengths, 95)}")
    print(f"99th percentile: {np.percentile(lengths, 99)}")
    print(f"Longer than {args.length}?: {np.mean(lengths > args.length):.2%}")
    print(f"Max length: {np.max(lengths)}")


if __name__ == "__main__":
    check_lengths(parse_args())
