import sys

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dblp-properties", action="store_true")
    return parser.parse_args()


def fix(s: str) -> str:
    return s.replace(r"\n", " ").replace(r"\t", " ")


def format(s: str) -> str:
    if s.startswith('"') and s.endswith('"'):
        # list of literals
        return fix(s[1:-1])
    elif s.startswith('"') and s.endswith('"@en'):
        # literal
        return fix(s[1:-4])
    else:
        return s


def get_dblp_label_from_prop_id(obj_id: str) -> str:
    prefix = "<https://dblp.org/rdf/schema#"
    assert obj_id.startswith(prefix)
    obj_name = obj_id[len(prefix):-1]
    assert len(obj_name.split()) == 1
    # split camelCase into words
    # find uppercase letters
    words = []
    last = 0
    for i, c in enumerate(obj_name):
        if c.isupper() and i > last:
            words.append(obj_name[last:i].lower())
            last = i

    if last < len(obj_name):
        words.append(obj_name[last:].lower())

    return " ".join(words)


if __name__ == "__main__":
    args = parse_args()
    header = next(sys.stdin)
    print("\t".join(
        field[1:] for field in
        header.rstrip("\r\n").split("\t")
    ))
    for line in sys.stdin:
        (
            label,
            score,
            syns,
            obj_id,
            infos
        ) = line.rstrip("\r\n").split("\t")

        label = format(label)
        syns = format(syns)
        infos = format(infos)
        if args.dblp_properties:
            # add the old label to syns
            syns = [s for s in syns.split(";;;") if s != ""]
            syns.append(label)
            syns = ";;;".join(syns)
            # replace with the new label
            label = get_dblp_label_from_prop_id(obj_id)

        score = "0" if score == "" else score

        print("\t".join([label, score, syns, obj_id, infos]))
