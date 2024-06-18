import re
import os
import pprint
import uuid
import collections
import copy
import requests
from importlib import resources
from typing import Any, Iterator

from tqdm import tqdm

from text_utils import text, grammar

QLEVER_API = "https://qlever.cs.uni-freiburg.de/api"
QLEVER_URLS = {
    "wikidata": f"{QLEVER_API}/wikidata",
    "dbpedia": f"{QLEVER_API}/dbpedia",
    "freebase": f"{QLEVER_API}/freebase",
}


class KgIndex:
    def __init__(
        self,
        index: dict[str, list[str]],
        redirect: dict[str, str] | None = None,
        prefixes: dict[str, str] | None = None
    ):
        self.index = index
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
            r"<?(" + "|".join(
                re.escape(long)
                for long in self.prefixes.values()
            ) + r")\w+>?"
        )

    @staticmethod
    def load(dir: str, progress: bool = False) -> "KgIndex":
        index_path = os.path.join(dir, "index.tsv")
        redirects_path = os.path.join(dir, "redirects.tsv")
        prefixes_path = os.path.join(dir, "prefixes.tsv")

        num_lines, _ = text.file_size(index_path)
        with open(index_path, "r", encoding="utf8") as f:
            index = {}
            for line in tqdm(
                f,
                total=num_lines,
                desc="loading kg index",
                disable=not progress,
                leave=False
            ):
                split = line.split("\t")
                assert len(split) >= 2
                obj = split[0].strip()
                obj_names = [n.strip() for n in split[1:]]
                assert obj not in index, \
                    f"duplicate id {obj}"
                index[obj] = obj_names

        redirect = {}
        if os.path.exists(redirects_path):
            num_lines, _ = text.file_size(redirects_path)
            with open(redirects_path, "r", encoding="utf8") as f:
                for line in tqdm(
                    f,
                    total=num_lines,
                    desc="loading kg redirects",
                    disable=not progress,
                    leave=False
                ):
                    split = line.split("\t")
                    assert len(split) >= 2
                    obj = split[0].strip()
                    for redir in split[1:]:
                        redir = redir.strip()
                        assert redir not in redirect, \
                            f"duplicate redirect {redir}, should not happen"
                        redirect[redir] = obj

        prefixes = {}
        if os.path.exists(prefixes_path):
            prefixes = load_prefixes(prefixes_path)

        return KgIndex(index, redirect, prefixes)

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
            return self.index[key]

        return default


def load_examples(path: str) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf8") as f:
        examples = []
        for line in f:
            split = line.split("\t")
            assert len(split) == 2
            query = split[0].strip()
            sparql = split[1].strip()
            examples.append((query, sparql))
        return examples


def load_prefixes(path: str) -> dict[str, str]:
    with open(path, "r", encoding="utf8") as f:
        prefixes = {}
        for line in f:
            split = line.split("\t")
            assert len(split) == 2
            short = split[0].strip()
            full = split[1].strip()
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


def _parse_to_string(parse: dict) -> str:
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
    indent: int = 2
) -> str:
    parse = parser.parse(sparql, skip_empty=True, collapse_single=False)

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
    return s


def _find_with_name(
    parse: dict,
    name: str,
    skip: set | None = None
) -> dict | None:
    if skip is not None and parse["name"] in skip:
        return None
    if parse["name"] == name:
        return parse
    for child in parse.get("children", []):
        t = _find_with_name(child, name)
        if t is not None:
            return t
    return None


def _find_all_with_name(
    parse: dict,
    name: str,
    skip: set | None = None
) -> Iterator[dict]:
    if skip is not None and parse["name"] in skip:
        return
    if parse["name"] == name:
        yield parse
        return
    for child in parse.get("children", []):
        yield from _find_all_with_name(child, name)


_CLEAN_PATTERN = re.compile(r"\s+", flags=re.MULTILINE)


def clean(s: str) -> str:
    return _CLEAN_PATTERN.sub(" ", s).strip()


def preprocess_natural_language_query(
    query: str,
    kgs: list[str],
    info: str | None,
    examples: list[tuple[str, str]] | None
) -> str:
    if len(kgs) == 0:
        kgs = ["None specified"]

    if examples is None or len(examples) == 0:
        example_list = ""
    else:
        example_list = "\n"

        for i, (query, sparql) in enumerate(examples):
            if len(examples) > 1:
                example_list += f"{i+1}. "
            example_list += f"Example:\n{clean(query)}\n{clean(sparql)}"
            if i < len(examples) - 1:
                example_list += "\n\n"

        example_list += "\n"

    if info is None or info.strip() == "":
        info = ""
    else:
        info = f"\nAdditional information / guidance:\n{info}\n"

    kg_list = "\n".join(kgs)
    return f"""\
Task:
SPARQL query generation over the specified knowledge graphs given a natural \
language query, optional additional information / guidance, and optional \
examples of query and SPARQL pairs.

Knowledge graphs:
{kg_list}

Query:
{query}
{info}{example_list}
SPARQL:
"""


_KG_PATTERN = re.compile("^<kg(?:e|p) kg='(\\w*)'>(.*?)</kg(?:e|p)>$")


def postprocess_sparql_query(
    sparql: str,
    parser: grammar.LR1Parser,
    entities: dict[str, dict[str, str]],
    properties: dict[str, dict[str, str]],
    prefixes: dict[str, str],
    pretty: bool = False,
) -> str:
    try:
        parse = parser.parse(sparql, skip_empty=True, collapse_single=True)
    except RuntimeError:
        return sparql

    for entity in _find_all_with_name(parse, "KGE"):
        val = entity["value"]
        kg_match = _KG_PATTERN.search(val)
        assert kg_match is not None
        kg = kg_match.group(1)
        if kg not in entities:
            continue
        kg_entities = entities[kg]
        value = kg_match.group(2)
        if value not in kg_entities:
            continue
        entity["value"] = kg_entities[value]

    for prop in _find_all_with_name(parse, "KGP"):
        val = prop["value"]
        kg_match = _KG_PATTERN.search(val)
        assert kg_match is not None
        kg = kg_match.group(1)
        if kg not in properties:
            continue
        kg_properties = properties[kg]
        value = kg_match.group(2)
        if value not in kg_properties:
            continue
        prop["value"] = kg_properties[value]

    sparql = _parse_to_string(parse)
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


class SelectResult:
    def __init__(
        self,
        vars: list[str],
        results: list[dict[str, SelectRecord | None]]
    ):
        self.vars = vars
        self.results = results

    def __len__(self) -> int:
        return len(self.results)

    def __repr__(self) -> str:
        return pprint.pformat(
            self.results,
            compact=True
        )


def ask_to_select(
    sparql: str,
    parser: grammar.LR1Parser,
    var: str | None = None,
    distinct: bool = False
) -> str | None:
    parse = parser.parse(sparql, skip_empty=False, collapse_single=False)

    sub_parse = _find_with_name(parse, "QueryType")
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
                _find_all_with_name(sub_parse, "Var", skip={"SubSelect"})
            ),
            None
        )
        assert ask_var is not None, "could not find specified var"
    else:
        ask_var = _find_with_name(sub_parse, "Var", skip={"SubSelect"})

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
        return _parse_to_string(parse)

    # ask query does not have a var, convert to select
    # and introduce own var
    # generate unique var name with uuid
    var = f"?{uuid.uuid4().hex}"
    iri = _find_with_name(sub_parse, "iri", skip={"SubSelect"})
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
    group_graph_pattern = _find_with_name(
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

    return _parse_to_string(parse)


def query_qlever(
    sparql_query: str,
    parser: grammar.LR1Parser,
    kg: str,
    qlever_endpoint: str | None
) -> SelectResult | AskResult:
    # ask_to_select return None if sparql is not an ask query
    select_query = ask_to_select(sparql_query, parser)

    sparql_query = select_query or sparql_query

    if qlever_endpoint is None:
        assert kg in QLEVER_URLS, \
            f"no QLever endpoint for knowledge graph {kg}"
        qlever_endpoint = QLEVER_URLS[kg]

    response = requests.post(
        qlever_endpoint,
        headers={"Content-type": "application/sparql-query"},
        data=sparql_query
    )
    json = response.json()

    if response.status_code != 200:
        msg = json.get("exception", "unknown exception")
        raise RuntimeError(
            f"query {sparql_query} returned with "
            f"status code {response.status_code}:\n{msg}"
        )

    if select_query is not None:
        return AskResult(len(json["results"]["bindings"]) > 0)

    vars = json["head"]["vars"]
    results = []
    for binding in json["results"]["bindings"]:
        result = {}
        for var in vars:
            if binding is None or var not in binding:
                result[var] = None
                continue
            value = binding[var]
            result[var] = SelectRecord(
                value["value"],
                value["type"]
            )
        results.append(result)
    return SelectResult(vars, results)


def query_entities(
    sparql: str,
    parser: grammar.LR1Parser,
    kg: str = "wikidata",
    qlever_endpoint: str | None = None
) -> set[tuple[str, ...]] | None:
    if qlever_endpoint is None:
        assert kg in QLEVER_URLS, \
            f"no QLever endpoint for knowledge graph {kg}"
        qlever_endpoint = QLEVER_URLS[kg]

    try:
        result = query_qlever(sparql, parser, kg, qlever_endpoint)
        if isinstance(result, AskResult):
            return {(f"{result}",)}
        return set(
            tuple(
                "" if r[var] is None else r[var].value  # type: ignore
                for var in result.vars
            )
            for r in result.results
        )
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
    tp = len(pred_set.intersection(target_set))
    fp = len(pred_set.difference(target_set))
    fn = len(target_set.difference(pred_set))
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

    prologue = _find_with_name(parse, "Prologue")
    assert prologue is not None

    base_decls = list(_find_all_with_name(prologue, "BaseDecl"))

    exist = {}
    for prefix_decl in _find_all_with_name(prologue, "PrefixDecl"):
        assert len(prefix_decl["children"]) == 3
        short = prefix_decl["children"][1]["value"]
        long = prefix_decl["children"][2]["value"][1:-1]
        exist[short] = long

    seen = set()
    for iri in _find_all_with_name(parse, "IRIREF"):
        value = iri["value"]
        val = value[1:-1]
        for short, long in prefixes.items():
            if val.startswith(long):
                iri["value"] = short + val[len(long):]
                seen.add(short)
                break

    for prefix_name in _find_all_with_name(parse, "PNAME_LN"):
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

    return _parse_to_string(parse)


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
    for var in _find_all_with_name(parse, "VAR1"):
        var["value"] = f"<bov>{var['value'][1:]}<eov>"

    for var in _find_all_with_name(parse, "VAR2"):
        var["value"] = f"<bov>{var['value'][1:]}<eov>"

    # replace brackets {, and } with <bob> and <eob>
    for bracket in _find_all_with_name(parse, "{"):
        bracket["value"] = "<bob>"

    for bracket in _find_all_with_name(parse, "}"):
        bracket["value"] = "<eob>"

    return _parse_to_string(parse)


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


def replace_entities_and_properties(
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

    entity_replacements = {
        kg: collections.Counter()
        for kg in entity_indices
    }
    property_replacements = {
        kg: collections.Counter()
        for kg in property_indices
    }

    def _replace_obj(
        obj: str,
        indices: dict[str, KgIndex],
        replacements: dict[str, collections.Counter],
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

            idx = replacements[kg][key]
            if idx < len(objs):
                replacements[kg][key] += 1
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
        for obj in _find_all_with_name(parse, "iri"):
            child = obj["children"][0]
            if child["name"] == "PrefixedName":
                val = child["children"][0]["value"]
            else:
                assert child["name"] == "IRIREF"
                val = child["value"]

            empty = False

            rep = _replace_obj(val, entity_indices, entity_replacements, True)
            # if rep is unchanged, this means that the val is not supported
            # by any of the entity indices
            if rep == val:
                rep = _replace_obj(val, property_indices,
                                   property_replacements, False)

            if rep is not None:
                obj["value"] = rep
                if rep == val:
                    ident = True
            else:
                incomplete = True

        return parse, incomplete, empty or ident

    parse, incomplete, _ = _replace_objs(parse)
    sparqls = [_parse_to_string(parse)]

    done = replacement == "only_first"
    while not done:
        parse, inc, stop = _replace_objs(copy.deepcopy(org_parse))
        done = inc or stop
        if not done:
            sparqls.append(_parse_to_string(parse))

    return sparqls, incomplete


_PREFIX_KG_PATTERN = re.compile(r"<kg(e|p) kg='(.*?)'>$")


def autocomplete_prefix(
    prefix: str,
    parser: grammar.LR1Parser,
    entities: dict[str, dict[str, str]],
    properties: dict[str, dict[str, str]],
    prefixes: dict[str, str],
    qlever_endpoint: str | None = None
) -> list[str] | None:
    """
    Autocomplete the SPARQL query prefix,
    run it against Qlever and return the entities or
    properties that can come next.
    Assumes that the prefix is a valid SPARQL query prefix.

    E.g. for "SELECT ?x WHERE { wd:Q5 <kgp kg='wikidata'>"
    the function would return ["wdt:P31", "wdt:P21", ...].
    """
    match = _PREFIX_KG_PATTERN.search(prefix)
    if match is None:
        return None

    prefix = prefix[:match.start()]
    obj_type = match.group(1)
    kg = match.group(2)

    if obj_type == "p":
        var = uuid.uuid4().hex
        obj_var = uuid.uuid4().hex
        prefix += f" ?{var} ?{obj_var} ."
    else:
        var = uuid.uuid4().hex
        prefix += f" ?{var} ."

    # keep track of brackets in the following stack
    brackets = []
    for (token, _) in parser.lex(prefix):
        if token == "{" or token == "(":
            brackets.append(token)
        elif token == "}" or token == ")":
            if len(brackets) == 0:
                raise RuntimeError("unbalanced brackets")
            last = brackets.pop()
            if token == "}" and last != "{":
                raise RuntimeError("unbalanced brackets")
            elif token == ")" and last != "(":
                raise RuntimeError("unbalanced brackets")

    # apply brackets in reverse order
    for bracket in reversed(brackets):
        if bracket == "{":
            prefix += " }"
        else:
            prefix += " )"

    select = ask_to_select(prefix, parser, var=f"?{var}", distinct=True)
    if select is not None:
        prefix = select
    else:
        # query is not an ask query, replace
        # the selected vars with our own
        parse = parser.parse(prefix, skip_empty=False, collapse_single=False)
        sel_clause = _find_with_name(parse, "SelectClause", skip={"SubSelect"})
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
        prefix = _parse_to_string(parse)

    prefix = fix_prefixes(
        prefix,
        parser,
        prefixes
    )
    result = query_qlever(
        prefix,
        parser,
        kg,
        qlever_endpoint
    )
    assert isinstance(result, SelectResult)
    next = []
    for r in result.results:
        record = r[var]
        if record is None:
            continue
        elif record.data_type != "uri":
            continue
        next.append(record.value)
    return next
