import argparse

from qgram_index import QGramIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("output", type=str, help="Output file")
    parser.add_argument("-q", "--qgram", type=int,
                        default=3, help="Qgram size")
    parser.add_argument("--no-syns", action="store_true",
                        help="Whether to remove synonyms")
    parser.add_argument("-d", "--distance",
                        choices=["ped", "ied"], default="ied")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    qgram_index = QGramIndex(args.q, not args.no_syns, args.distance)
    qgram_index.build(args.input)
    qgram_index.save(args.output)
