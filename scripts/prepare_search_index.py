import sys


def format(s: str) -> str:
    if s.startswith('"') and s.endswith('"'):
        # list of literals
        return s[1:-1]
    elif s.startswith('"') and s.endswith('"@en'):
        # literal
        return "".join(s[1:-4])
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
            count,
            syns,
            obj_id,
            *infos
        ) = line.rstrip("\r\n").split("\t")

        label = format(label)
        syns = format(syns)
        count = "0" if count == "" else count

        infos = [
            format(info)
            for info in infos
        ]
        print("\t".join([label, count, syns, obj_id] + infos))
