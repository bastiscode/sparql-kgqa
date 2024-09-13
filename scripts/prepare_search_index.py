import sys

from sparql_kgqa.sparql.utils2 import clean


def format(s: str) -> str:
    if s.startswith('"') and s.endswith('"'):
        # list of literals
        return clean(s[1:-1])
    elif s.startswith('"') and s.endswith('"@en'):
        # literal
        return clean(s[1:-4])
    else:
        return s


if __name__ == "__main__":
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
            *infos
        ) = line.rstrip("\r\n").split("\t")

        label = format(label)
        syns = format(syns)
        score = "0" if score == "" else score

        infos = [
            format(info) for info in infos
        ]
        print("\t".join([label, score, syns, obj_id] + infos))
