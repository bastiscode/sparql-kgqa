import argparse
import os

from qgram_index import QGramIndex

from sparql_kgqa.sparql.utils2 import Mapping


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
    parser.add_argument("--with-mapping", action="store_true",
                        help="Whether to compute and save the mapping")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    qgram_index = QGramIndex(args.qgram, not args.no_syns, args.distance)
    qgram_index.build(args.input)
    qgram_index.save(args.output)
    if args.with_mapping:
        mapping = Mapping()
        mapping.build_from_qgram_index(qgram_index)
        file, _ = os.path.splitext(args.output)
        mapping.save(file + ".mapping.json.bz2")
