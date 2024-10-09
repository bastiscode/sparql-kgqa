import argparse
import os

from search_index import QGramIndex, PrefixIndex
from search_index.mapping import Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("output", type=str, help="Output dir")
    parser.add_argument("--no-syns", action="store_true",
                        help="Whether to remove synonyms")
    parser.add_argument("-d", "--distance",
                        choices=["ped", "ied"], default="ied")
    parser.add_argument("--with-mapping", action="store_true",
                        help="Whether to compute and save the mapping")
    parser.add_argument("--mapping-column", type=int, default=3)
    parser.add_argument("--type", type=str, default="qgram",
                        choices=["qgram", "prefix"])
    parser.add_argument("-q", "--qgram", type=int,
                        default=3, help="Qgram size")
    return parser.parse_args()


def build(args: argparse.Namespace):
    print(f"Building {args.type} index")
    if args.type == "qgram":
        QGramIndex.build(
            args.input,
            args.output,
            args.qgram,
            args.distance,
            use_synonyms=not args.no_syns
        )
    else:
        PrefixIndex.build(
            args.input,
            args.output,
            use_synonyms=not args.no_syns
        )

    if not args.with_mapping:
        return

    if args.type == "qgram":
        index = QGramIndex.load(
            args.input,
            args.output,
        )
    else:
        index = PrefixIndex.load(
            args.input,
            args.output,
        )

    Mapping.build(
        index,
        os.path.join(args.output, "index.mapping"),
        args.mapping_column
    )


if __name__ == "__main__":
    build(parse_args())
