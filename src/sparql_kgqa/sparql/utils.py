import re
import json
import mmap
import os
import uuid
import collections
import copy
import requests
from collections import Counter
from importlib import resources
from typing import Any, Iterator

import torch
from tqdm import tqdm

from text_utils import grammar

QLEVER_API = "https://qlever.cs.uni-freiburg.de/api"
QLEVER_URLS = {
    "wikidata": f"{QLEVER_API}/wikidata",
    "dbpedia": f"{QLEVER_API}/dbpedia",
    "freebase": f"{QLEVER_API}/freebase",
    "dblp": f"{QLEVER_API}/dblp",
}


class KgIndex:
    def __init__(
        self,
        index: dict[str, tuple[int, int]],
        index_file: str,
        redirect: dict[str, str] | None = None,
        prefixes: dict[str, str] | None = None
    ):
        self.index = index
        self.index_file = open(index_file, "r+b")
        self.index_mmap = mmap.mmap(self.index_file.fileno(), 0)
        self.redirect = redirect or {}
        self.prefixes = prefixes or {}
        self.reverse_prefixes = {
            long: short
            for short, long in self.prefixes.items()
        }
        self.short_key_pattern = re.compile(
            r"(" + "|".join(
                re.escape(short)
                for short in self.prefixes
            ) + r")\w+"
        )
        self.long_key_pattern = re.compile(
            r"<?((:?" + "|".join(
                re.escape(long)
                for long in self.prefixes.values()
            ) + r")\w+)>?"
        )

    @staticmethod
    def load(dir: str, progress: bool = False) -> "KgIndex":
        index_path = os.path.join(dir, "index.tsv")
        redirects_path = os.path.join(dir, "redirects.tsv")
        prefixes_path = os.path.join(dir, "prefixes.tsv")

        with open(index_path, "r", encoding="utf8") as f:
            index = {}
            offset = 0
            for line in tqdm(
                f,
                desc="loading kg index",
                disable=not progress,
                leave=False
            ):
                line_length = len(line.encode())
                split = line.split("\t")
                assert len(split) >= 2
                obj = split[0].strip()
                index[obj] = (offset, line_length)
                offset += line_length

        redirect = {}
        if os.path.exists(redirects_path):
            with open(redirects_path, "r", encoding="utf8") as f:
                for line in tqdm(
                    f,
                    desc="loading kg redirects",
                    disable=not progress,
                    leave=False
                ):
                    split = line.rstrip("\r\n").split("\t")
                    assert len(split) >= 2
                    obj = split[0]
                    for redir in split[1:]:
                        assert redir not in redirect, \
                            f"duplicate redirect {redir}, should not happen"
                        redirect[redir] = obj

        prefixes = {}
        if os.path.exists(prefixes_path):
            prefixes = load_prefixes(prefixes_path)

        return KgIndex(index, index_path, redirect, prefixes)

    def normalize_key(
        self,
        key: str
    ) -> str | None:
        match = self.short_key_pattern.fullmatch(key)
        if match is not None:
            # translate short key to long key
            short = match.group(1)
            long = self.prefixes[short]
            return long + key[len(short):]

        match = self.long_key_pattern.fullmatch(key)
        if match is not None:
            return match.group(1)

        return None

    def get(
        self,
        key: str,
        default: list[str] | None = None
    ) -> list[str] | None:
        while key not in self.index and self.redirect.get(key, key) != key:
            key = self.redirect[key]

        if key in self.index:
            offset, length = self.index[key]
            line = self.index_mmap[offset:offset + length].decode()
            values = line.rstrip("\r\n").split("\t")[1:]
            return values

        return default


Examples = list[tuple[str, str]]


def load_examples(path: str) -> Examples:
    with open(path, "r", encoding="utf8") as f:
        examples = []
        for line in f:
            data = json.loads(line.rstrip("\r\n"))
            examples.append((data["query"], data["sparql"]))
        return examples


def load_prefixes(path: str) -> dict[str, str]:
    with open(path, "r", encoding="utf8") as f:
        prefixes = {}
        for line in f:
            split = line.rstrip("\r\n").split("\t")
            assert len(split) == 2
            short, full = split
            assert short not in prefixes, \
                f"duplicate prefix {short}"
            prefixes[short] = full
        return prefixes


def load_inverse_index(path: str) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf8") as f:
        index = {}
        for line in f:
            split = line.strip().split("\t")
            assert len(split) == 2
            obj_id_1 = split[0].strip()
            obj_id_2 = split[1].strip()
            if obj_id_1 not in index:
                index[obj_id_1] = [obj_id_2]
            else:
                index[obj_id_1].append(obj_id_2)
        return index


def general_prefixes() -> dict[str, str]:
    return {
        "bd:": "http://www.bigdata.com/rdf#",
        "cc:": "http://creativecommons.org/ns#",
        "dct:": "http://purl.org/dc/terms/",
        "geo:": "http://www.opengis.net/ont/geosparql#",
        "hint:": "http://www.bigdata.com/queryHints#",
        "ontolex:": "http://www.w3.org/ns/lemon/ontolex#",
        "owl:": "http://www.w3.org/2002/07/owl#",
        "prov:": "http://www.w3.org/ns/prov#",
        "rdf:": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs:": "http://www.w3.org/2000/01/rdf-schema#",
        "schema:": "http://schema.org/",
        "skos:": "http://www.w3.org/2004/02/skos/core#",
        "xsd:": "http://www.w3.org/2001/XMLSchema#",
        "wikibase:": "http://wikiba.se/ontology#",
    }


def _load_sparql_grammar(kgs: list[str]) -> tuple[str, str]:
    sparql_grammar = resources.read_text(
        "sparql_kgqa.sparql.grammar",
        "sparql.y"
    )
    sparql_lexer = resources.read_text(
        "sparql_kgqa.sparql.grammar",
        "sparql.l"
    )
    if len(kgs) > 0:
        sparql_lexer = sparql_lexer.replace(
            "KNOWLEDGE_GRAPHS ''",
            f"KNOWLEDGE_GRAPHS {'|'.join(re.escape(kg) for kg in kgs)}"
        )
    return sparql_grammar, sparql_lexer


def load_sparql_parser(kgs: list[str]) -> grammar.LR1Parser:
    return grammar.LR1Parser(*_load_sparql_grammar(kgs))


def load_sparql_constraint(
    kgs: list[str],
    continuations: list[bytes],
    exact: bool
) -> grammar.LR1Constraint:
    return grammar.LR1Constraint(
        *_load_sparql_grammar(kgs),
        continuations,
        exact
    )


def parse_to_string(parse: dict) -> str:
    def _flatten(parse: dict) -> str:
        if "value" in parse:
            return parse["value"]
        elif "children" in parse:
            children = []
            for p in parse["children"]:
                child = _flatten(p)
                if child != "":
                    children.append(child)
            return " ".join(children)
        else:
            return ""

    return _flatten(parse)


def prettify(
    sparql: str,
    parser: grammar.LR1Parser,
    indent: int = 4,
    is_prefix: bool = False
) -> str:
    if is_prefix:
        parse, rest = parser.prefix_parse(
            sparql.encode(),
            skip_empty=True,
            collapse_single=False
        )
        rest_str = bytes(rest).decode(errors="replace")
    else:
        parse = parser.parse(
            sparql,
            skip_empty=True,
            collapse_single=False
        )
        rest_str = ""

    # some simple rules for pretty printing:
    # 1. new lines after prologue (PrologueDecl) and triple blocks
    # (TriplesBlock)
    # 2. new lines after { and before }
    # 3. increase indent after { and decrease before }

    assert indent > 0, "indent step must be positive"
    current_indent = 0
    s = ""

    def _pretty(parse: dict) -> bool:
        nonlocal current_indent
        nonlocal s
        newline = False

        if "value" in parse:
            if parse["name"] in [
                "UNION",
                "MINUS"
            ]:
                s = s.rstrip() + " "

            elif parse["name"] == "}":
                current_indent -= indent
                s = s.rstrip()
                s += "\n" + " " * current_indent

            elif parse["name"] == "{":
                current_indent += indent

            s += parse["value"]

        elif len(parse["children"]) == 1:
            newline = _pretty(parse["children"][0])

        else:
            for i, child in enumerate(parse["children"]):
                if i > 0 and not newline and child["name"] != "(":
                    s += " "

                newline = _pretty(child)

        if not newline and parse["name"] in [
            "{",
            "}",
            ".",
            "PrefixDecl",
            "BaseDecl",
            "TriplesBlock",
            "GroupClause",
            "HavingClause",
            "OrderClause",
            "LimitClause",
            "OffsetClause",
            "GraphPatternNotTriples"
        ]:
            s += "\n" + " " * current_indent
            newline = True

        return newline

    newline = _pretty(parse)
    if newline:
        s = s.rstrip()
    return (s.strip() + " " + rest_str).strip()


def find(
    parse: dict,
    name: str | set[str],
    skip: set[str] | None = None
) -> dict | None:
    return next(find_all(parse, name, skip), None)


def find_all(
    parse: dict,
    name: str | set[str],
    skip: set[str] | None = None
) -> Iterator[dict]:
    if skip is not None and parse["name"] in skip:
        return
    elif isinstance(name, str) and parse["name"] == name:
        yield parse
    elif isinstance(name, set) and parse["name"] in name:
        yield parse
    else:
        for child in parse.get("children", []):
            yield from find_all(child, name, skip)


_CLEAN_PATTERN = re.compile(r"\s+", flags=re.MULTILINE)


def clean(s: str) -> str:
    return _CLEAN_PATTERN.sub(" ", s).strip()


def preprocess_natural_language_query(
    query: str,
    kgs: list[str],
    examples: list[tuple[str, str]] | None = None
) -> str:
    if len(kgs) == 0:
        kg_str = "None"
    else:
        kg_str = "\n".join(kgs)

    if examples is None or len(examples) == 0:
        example_str = "None"
    else:
        example_str = "\n\n".join(
            f"{i+1}. Example:\n{clean(query)}\n{clean(sparql)}"
            for i, (query, sparql) in enumerate(examples)
        )

    return f"""\
Task:
SPARQL query generation over the specified knowledge graphs given a natural \
language query and optional examples of query and SPARQL pairs.

Knowledge graphs:
{kg_str}

Query:
{query}

Examples:
{example_str}

SPARQL:
"""


_KG_PATTERN = re.compile("<kg(e|p) kg='(\\w*)'>(.*?)</kg\\1>")


def replace_entities_and_properties(
    sparql: str | dict,
    parser: grammar.LR1Parser,
    entities: dict[str, dict[str, str]],
    properties: dict[str, dict[str, str]],
) -> str:
    if isinstance(sparql, str):
        try:
            parse = parser.parse(
                sparql,
                skip_empty=True,
                collapse_single=True
            )
        except RuntimeError:
            return sparql
    else:
        parse = sparql

    def _find_and_replace(name: str, objects: dict[str, dict[str, str]]):
        for obj in find_all(parse, name):
            if "value" not in obj:
                continue
            kg_match = _KG_PATTERN.search(obj["value"])
            if kg_match is None:
                continue
            kg = kg_match.group(2)
            if kg not in objects:
                continue
            kg_objects = objects[kg]
            value = kg_match.group(3)
            if value not in kg_objects:
                continue
            obj["value"] = "<" + kg_objects[value] + ">"

    _find_and_replace("KGE", entities)
    _find_and_replace("KGP", properties)
    return parse_to_string(parse)


def postprocess_sparql_query(
    sparql: str,
    parser: grammar.LR1Parser,
    entities: dict[str, dict[str, str]],
    properties: dict[str, dict[str, str]],
    prefixes: dict[str, str],
    pretty: bool = False,
) -> str:
    sparql = replace_entities_and_properties(
        sparql,
        parser,
        entities,
        properties
    )
    sparql = fix_prefixes(sparql, parser, prefixes)
    if pretty:
        sparql = prettify(sparql, parser)
    return sparql


class SelectRecord:
    def __init__(
        self,
        value: str,
        data_type: str,
        label: str | None = None
    ):
        self.value = value
        self.data_type = data_type
        self.label = label

    def __repr__(self) -> str:
        return self.label or self.value


AskResult = bool
SelectResult = list[list[str]]

# class SelectResult:
#     def __init__(
#         self,
#         vars: list[str],
#         results: list[dict[str, SelectRecord | None]]
#     ):
#         self.vars = vars
#         self.results = results
#
#     def __len__(self) -> int:
#         return len(self.results)
#
#     def __repr__(self) -> str:
#         return pprint.pformat(
#             self.results,
#             compact=True
#         )


def ask_to_select(
    sparql: str,
    parser: grammar.LR1Parser,
    var: str | None = None,
    distinct: bool = False
) -> str | None:
    parse = parser.parse(sparql, skip_empty=False, collapse_single=False)

    sub_parse = find(parse, "QueryType")
    assert sub_parse is not None

    ask_query = sub_parse["children"][0]
    if ask_query["name"] != "AskQuery":
        return None

    # we have an ask query
    # find the first var or iri
    if var is not None:
        ask_var = next(
            filter(
                lambda p: p["children"][0]["value"] == var,
                find_all(sub_parse, "Var", skip={"SubSelect"})
            ),
            None
        )
        assert ask_var is not None, "could not find specified var"
    else:
        ask_var = find(sub_parse, "Var", skip={"SubSelect"})

    if ask_var is not None:
        var = ask_var["children"][0]["value"]
        # ask query has a var, convert to select
        ask_query["name"] = "SelectQuery"
        # replace ASK terminal with SelectClause
        sel_clause: list[dict[str, Any]] = [{
            'name': 'SELECT',
            'value': 'SELECT',
        }]
        if distinct:
            sel_clause.append({
                'name': 'DISTINCT',
                'value': 'DISTINCT'
            })
        sel_clause.append({
            'name': 'Var',
            'value': var
        })
        ask_query["children"][0] = {
            'name': 'SelectClause',
            'children': sel_clause
        }
        return parse_to_string(parse)

    # ask query does not have a var, convert to select
    # and introduce own var
    # generate unique var name with uuid
    var = f"?{uuid.uuid4().hex}"
    iri = find(sub_parse, "iri", skip={"SubSelect"})
    assert iri is not None, "could not find an iri in ask query"

    child = iri["children"][0]
    if child["name"] == "PrefixedName":
        iri = child["children"][0]["value"]
        child["children"][0]["value"] = var
    elif child["name"] == "IRIREF":
        iri = child["value"]
        child["value"] = var
    else:
        raise ValueError(f"unsupported iri format {iri}")

    where_clause = ask_query["children"][2]
    group_graph_pattern = find(
        where_clause,
        "GroupGraphPattern",
        skip={"SubSelect"}
    )
    assert group_graph_pattern is not None
    values = {
        "name": "CustomValuesClause",
        "value": f"VALUES {var} {{ {iri} }}"
    }
    group_graph_pattern["children"].insert(1, values)

    ask_query["name"] = "SelectQuery"
    # replace ASK terminal with SelectClause
    ask_query["children"][0] = {
        'name': 'SelectClause',
        'children': [
            {
                'name': 'SELECT',
                'value': 'SELECT',
            },
            {
                'name': 'Var',
                'value': var
            }
        ],
    }

    return parse_to_string(parse)


def query_qlever(
    sparql_query: str,
    parser: grammar.LR1Parser,
    kg: str,
    qlever_endpoint: str | None,
    timeout: float | tuple[float, float] | None = None,
    max_retries: int = 1
) -> SelectResult | AskResult:
    # ask_to_select return None if sparql is not an ask query
    select_query = ask_to_select(sparql_query, parser)

    sparql_query = select_query or sparql_query

    if qlever_endpoint is None:
        assert kg in QLEVER_URLS, \
            f"no QLever endpoint for knowledge graph {kg}"
        qlever_endpoint = QLEVER_URLS[kg]

    response = None
    for _ in range(max(1, max_retries)):
        response = requests.post(
            qlever_endpoint,
            headers={
                "Content-type": "application/sparql-query",
                "Accept": "text/tab-separated-values"
            },
            data=sparql_query,
            timeout=timeout
        )
        if response.status_code == 200:
            break

    assert response is not None, "cannot happen"

    if response.status_code != 200:
        exception = response.json().get("exception", "")
        raise RuntimeError(exception)

    result = response.text.splitlines()
    if select_query is not None:
        # > 1 because of header
        return AskResult(len(result) > 1)
    else:
        return [row.split("\t") for row in result]


def query_entities(
    sparql: str,
    parser: grammar.LR1Parser,
    kg: str = "wikidata",
    qlever_endpoint: str | None = None
) -> Counter | None:
    if qlever_endpoint is None:
        assert kg in QLEVER_URLS, \
            f"no QLever endpoint for knowledge graph {kg}"
        qlever_endpoint = QLEVER_URLS[kg]

    try:
        result = query_qlever(
            sparql,
            parser,
            kg,
            qlever_endpoint,
            timeout=(5.0, 60.0)
        )
        if isinstance(result, AskResult):
            return Counter({result: 1})
        else:
            return Counter((tuple(r) for r in result))
    except Exception:
        return None


def calc_f1(
    pred: str,
    target: str,
    parser: grammar.LR1Parser,
    allow_empty_target: bool = True,
    kg: str = "wikidata",
    qlever_endpoint: str | None = None
) -> tuple[float | None, bool, bool]:
    pred_set = query_entities(pred, parser, kg, qlever_endpoint)
    target_set = query_entities(target, parser, kg, qlever_endpoint)
    if pred_set is None or target_set is None:
        return None, pred_set is None, target_set is None
    if len(target_set) == 0 and not allow_empty_target:
        return None, False, True
    if len(pred_set) == 0 and len(target_set) == 0:
        return 1.0, False, False
    tp = (pred_set & target_set).total()
    fp = (pred_set - target_set).total()
    fn = (target_set - pred_set).total()
    # calculate precision, recall and f1
    if tp > 0:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0.0
    return f1, False, False


def fix_prefixes(
    sparql: str,
    parser: grammar.LR1Parser,
    prefixes: dict[str, str]
) -> str:
    """
    Clean the prefixes in the SPARQL query.

    >>> parser = grammar.LR1Parser(*_load_sparql_grammar([]))
    >>> prefixes = {"test:": "http://test.com/"}
    >>> s = "PREFIX bla: <unrelated> SELECT ?x WHERE { ?x \
    <http://test.de/prop> ?y }"
    >>> fix_prefixes(s, parser, prefixes)
    'SELECT ?x WHERE { ?x <http://test.de/prop> ?y }'
    >>> s = "SELECT ?x WHERE { ?x test:prop ?y }"
    >>> fix_prefixes(s, parser, prefixes)
    'PREFIX test: <http://test.com/> SELECT ?x WHERE { ?x test:prop ?y }'
    >>> s = "SELECT ?x WHERE { ?x <http://test.com/prop> ?y }"
    >>> fix_prefixes(s, parser, prefixes)
    'PREFIX test: <http://test.com/> SELECT ?x WHERE { ?x test:prop ?y }'
    """
    parse = parser.parse(sparql, skip_empty=False, collapse_single=True)

    prologue = find(parse, "Prologue")
    assert prologue is not None

    base_decls = list(find_all(prologue, "BaseDecl"))

    exist = {}
    for prefix_decl in find_all(prologue, "PrefixDecl"):
        assert len(prefix_decl["children"]) == 3
        short = prefix_decl["children"][1]["value"]
        long = prefix_decl["children"][2]["value"][1:-1]
        exist[short] = long

    seen = set()
    for iri in find_all(parse, "IRIREF"):
        value = iri["value"]
        val = value[1:-1]

        longest: tuple[str, str] | None = next(iter(sorted(
            filter(
                lambda pair: val.startswith(pair[1]),
                prefixes.items()
            ),
            key=lambda pair: len(pair[1]),
            reverse=True
        )), None)
        if longest is None:
            continue

        short, long = longest
        iri["value"] = short + val[len(long):]
        seen.add(short)

    for prefix_name in find_all(parse, "PNAME_LN"):
        val = prefix_name["value"]
        val = val[:val.find(":") + 1]
        seen.add(val)

    prologue["children"] = base_decls

    for prefix in seen:
        if prefix in exist:
            long = exist[prefix]
        elif prefix in prefixes:
            long = prefixes[prefix]
        else:
            continue
        prologue["children"].append(
            {
                "name": "PrefixDecl",
                "children": [
                    {"name": "PREFIX", "value": "PREFIX"},
                    {"name": "PNAME_NS", "value": f"{prefix}"},
                    {"name": "IRIREF", "value": f"<{long}>"},
                ]
            }
        )

    return parse_to_string(parse)


def replace_vars_and_special_tokens(
    sparql: str,
    parser: grammar.LR1Parser,
    version: str,
) -> str:
    """
    Replace variables and special tokens in the SPARQL query.

    >>> parser = grammar.LR1Parser(*_load_sparql_grammar([]))
    >>> s = "SELECT ?x WHERE { ?x ?y $z }"
    >>> replace_vars_and_special_tokens(s, parser, "v2")
    'SELECT ?x WHERE { ?x ?y $z }'
    >>> replace_vars_and_special_tokens(s, parser, "v1")
    'SELECT <bov>x<eov> WHERE <bob> <bov>x<eov> <bov>y<eov> <bov>z<eov> <eob>'
    """
    if version != "v1":
        return sparql

    parse = parser.parse(sparql, skip_empty=True, collapse_single=True)
    for var in find_all(parse, "VAR1"):
        var["value"] = f"<bov>{var['value'][1:]}<eov>"

    for var in find_all(parse, "VAR2"):
        var["value"] = f"<bov>{var['value'][1:]}<eov>"

    # replace brackets {, and } with <bob> and <eob>
    for bracket in find_all(parse, "{"):
        bracket["value"] = "<bob>"

    for bracket in find_all(parse, "}"):
        bracket["value"] = "<eob>"

    return parse_to_string(parse)


def format_ent(ent: str, version: str, kg: str) -> str:
    if version == "v2":
        return f"<kge kg='{kg}'>{ent}</kge>"
    else:
        return f"<boe>{ent}<eoe>"


def format_prop(prop: str, version: str, kg: str) -> str:
    if version == "v2":
        return f"<kgp kg='{kg}'>{prop}</kgp>"
    else:
        return f"<bop>{prop}<eop>"


def replace_iris(
    sparql: str,
    parser: grammar.LR1Parser,
    entity_indices: dict[str, KgIndex],
    property_indices: dict[str, KgIndex],
    version: str,
    replacement: str = "only_first",
) -> tuple[list[str], bool]:
    assert replacement in [
        "only_first",
        "in_order"
    ]

    parse = parser.parse(sparql, skip_empty=True, collapse_single=False)
    org_parse = copy.deepcopy(parse)

    ent_off = {
        kg: collections.Counter()
        for kg in entity_indices
    }
    prop_off = {
        kg: collections.Counter()
        for kg in property_indices
    }

    def _replace_obj(
        obj: str,
        indices: dict[str, KgIndex],
        offsets: dict[str, collections.Counter],
        replacements: set[tuple[str, str]],
        is_ent: bool
    ) -> str | None:
        nkeys = {}
        for kg, index in indices.items():
            nkey = index.normalize_key(obj)
            if nkey is not None:
                nkeys[kg] = nkey

        # no index supports the key, this is different
        # from the key being supported but not found
        if len(nkeys) == 0:
            return obj

        for kg, key in nkeys.items():
            index = indices[kg]

            objs = index.get(key)
            if objs is None:
                continue

            idx = offsets[kg][key]
            if idx < len(objs):
                replacements.add((kg, key))
                if is_ent:
                    return format_ent(objs[idx], version, kg)
                else:
                    return format_prop(objs[idx], version, kg)

        # in none of the supported indices was the key found
        return None

    def _replace_objs(parse: dict) -> tuple[dict, bool, bool]:
        empty = True
        ident = False
        incomplete = False
        ent_rep = set()
        prop_rep = set()
        for obj in find_all(parse, "iri"):
            child = obj["children"][0]
            if child["name"] == "PrefixedName":
                val = child["children"][0]["value"]
            else:
                assert child["name"] == "IRIREF"
                val = child["value"]

            empty = False

            rep = _replace_obj(val, entity_indices, ent_off, ent_rep, True)
            # if rep is unchanged, this means that the val is not supported
            # by any of the entity indices
            if rep == val:
                rep = _replace_obj(
                    val,
                    property_indices,
                    prop_off,
                    prop_rep,
                    False
                )

            if rep is not None:
                obj["value"] = rep
                if rep == val:
                    ident = True
            else:
                incomplete = True

        for (kg, rep) in ent_rep:
            ent_off[kg][rep] += 1

        for (kg, rep) in prop_rep:
            prop_off[kg][rep] += 1

        return parse, incomplete, empty or ident

    parse, incomplete, _ = _replace_objs(parse)
    sparqls = [parse_to_string(parse)]

    done = replacement == "only_first"
    while not done:
        parse, inc, stop = _replace_objs(copy.deepcopy(org_parse))
        done = inc or stop
        if not done:
            sparqls.append(parse_to_string(parse))

    return sparqls, incomplete


_KG_PREFIX_PATTERN = re.compile(r"<kg(?:e|p) kg='(.*?)'>")


def subgraph_constraint(
    prefix: str,
    parser: grammar.LR1Parser,
    entities: dict[str, dict[str, str]],
    properties: dict[str, dict[str, str]],
    prefixes: dict[str, str],
    qlever_endpoint: str | None = None,
    limit: int | None = None
) -> list[str] | None:
    """
    Autocomplete the SPARQL query prefix,
    run it against Qlever and return the entities or
    properties that can come next.
    Assumes that the prefix is a valid SPARQL query prefix.

    E.g. for "SELECT ?x WHERE { wd:Q5 <kgp kg='wikidata'>"
    the function would return ["wdt:P31", "wdt:P21", ...].
    """
    matches = list(_KG_PREFIX_PATTERN.finditer(prefix))
    if len(matches) == 0 or matches[-1].end() != len(prefix):
        return None

    match = matches[-1]
    prefix = prefix[:match.start()]
    kg = match.group(1)

    parse, _ = parser.prefix_parse(
        prefix.encode(),
        skip_empty=False,
        collapse_single=True
    )

    # determine current position in the query:
    # subject, predicate or object
    triple_blocks = list(find_all(
        parse,
        "TriplesSameSubjectPath",
    ))
    if len(triple_blocks) == 0:
        # without triples the knowledge graph can not be
        # constrained
        return None

    last_triple = triple_blocks[-1]
    # the last triple block
    assert len(last_triple["children"]) == 2
    first, second = last_triple["children"]
    if second["name"] != "PropertyListPathNotEmpty":
        return None

    var = uuid.uuid4().hex
    assert len(second["children"]) == 3
    if parse_to_string(second["children"][1]) != "":
        # subject can always be any iri
        return None

    elif parse_to_string(second["children"][0]) != "":
        # object
        second["children"][1] = {"name": "VAR1", "value": f"?{var}"}

    elif parse_to_string(first) != "":
        # property
        second["children"][0] = {"name": "VAR1", "value": f"?{var}"}
        obj_var = uuid.uuid4().hex
        second["children"][1] = {"name": "VAR1", "value": f"?{obj_var}"}

    else:
        # unexpected case
        return None

    if find(
        parse,
        {
            "PNAME_LN",
            "IRIREF",
            "KGE",
            "KGP"
        },
        skip={"SubSelect"}
    ) is None:
        # means we have a prefix but with only vars,
        # which is not useful for autocomplete since only
        # iris constrain the knowledge graph
        return None

    # fix all future brackets
    for item in find_all(
        parse,
        {"{", "}", "(", ")", "."},
    ):
        item["value"] = item["name"]

    prefix = parse_to_string(parse)

    select = ask_to_select(prefix, parser, var=f"?{var}", distinct=True)
    if select is not None:
        prefix = select
    else:
        # query is not an ask query, replace
        # the selected vars with our own
        parse = parser.parse(prefix, skip_empty=False, collapse_single=False)
        sel_clause = find(parse, "SelectClause", skip={"SubSelect"})
        assert sel_clause is not None, "could not find select clause"
        sel_clause["children"] = [
            {
                'name': 'SELECT',
                'value': 'SELECT',
            },
            {
                'name': "DISTINCT",
                'value': "DISTINCT"
            },
            {
                'name': 'Var',
                'value': f"?{var}"
            }
        ]
        prefix = parse_to_string(parse)

    if limit is not None:
        # find solution modifier and add limit clause
        parse = parser.parse(
            prefix,
            skip_empty=False,
            collapse_single=False
        )
        limit_clause = find(
            parse,
            "LimitOffsetClausesOptional",
            skip={"SubSelect"}
        )
        assert limit_clause is not None, \
            "could not find limit clause"
        assert "children" not in limit_clause, \
            "limit clause should be empty"
        limit_clause["children"] = [{
            "name": "LimitClause",
            "value": f"LIMIT {limit}"
        }]
        prefix = parse_to_string(parse)

    prefix = postprocess_sparql_query(
        prefix,
        parser,
        entities,
        properties,
        prefixes
    )
    result = query_qlever(
        prefix,
        parser,
        kg,
        qlever_endpoint,
        timeout=(1.0, 3.0)
    )
    assert isinstance(result, list)
    return [r[0] for r in result]


class SimilarityIndex:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.dim = 768
        self.embeddings = torch.zeros(0, self.dim, dtype=torch.float)
        self.data = []
        self.seen = set()

    def add(
        self,
        examples: Examples,
        batch_size: int = 32,
        show_progress: bool = False
    ):
        unseen_examples = []
        for query, sparql in examples:
            sample = (query.lower(), sparql)
            if sample in self.seen:
                continue
            unseen_examples.append(sample)
            self.seen.add(sample)

        embeddings = self.model.encode(
            [query for query, _ in unseen_examples],
            batch_size=batch_size,
            convert_to_numpy=False,
            convert_to_tensor=True,
            show_progress_bar=show_progress
        )
        assert isinstance(embeddings, torch.Tensor)
        self.embeddings = torch.cat([
            self.embeddings,
            embeddings.to(self.embeddings.device)
        ])
        self.data.extend(unseen_examples)
        assert len(self.data) == self.embeddings.shape[0] \
            and len(self.data) == len(self.seen)

    def save(self, path: str):
        torch.save(
            {
                "embeddings": self.embeddings,
                "data": self.data
            },
            path
        )

    @staticmethod
    def load(path: str) -> "SimilarityIndex":
        index = SimilarityIndex()
        data = torch.load(path)
        index.embeddings = data["embeddings"]
        index.data = data["data"]
        index.seen = set(index.data)
        return index

    def top_k(
        self,
        query: str | list[str],
        k: int = 5,
        batch_size: int = 32
    ) -> Examples | list[Examples]:
        is_batched = True
        if isinstance(query, str):
            is_batched = False
            query = [query]

        # lowercase to be consistent with the embeddings
        query = [q.lower() for q in query]

        query_embedding = self.model.encode(
            query,
            batch_size=batch_size,
            convert_to_numpy=False,
            convert_to_tensor=True
        )
        assert isinstance(query_embedding, torch.Tensor)
        similarities = self.model.similarity(
            query_embedding.to(self.embeddings.device),
            self.embeddings
        )
        top_k = torch.topk(similarities, min(k, similarities.shape[-1]))
        top_k_samples = [
            [self.data[i] for i in indices]
            for indices in top_k.indices.tolist()
        ]
        if is_batched:
            return top_k_samples
        else:
            return top_k_samples[0]
