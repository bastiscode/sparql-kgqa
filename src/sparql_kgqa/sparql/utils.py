import re
import collections
import copy
import requests
from importlib import resources
from typing import Iterator

from tqdm import tqdm

from text_utils import text, grammar, continuations
from text_utils.api.table import generate_table

ContIndex = continuations.ContinuationIndex

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
            "|".join(
                rf"{re.escape(short)}\w+"
                for short in self.prefixes
            )
        )
        self.long_key_pattern = re.compile(
            "|".join(
                rf"<(?P<long{i}>"
                + re.escape(long)
                + rf")(?P<long{i}_>\w+)>"
                for i, long in enumerate(self.prefixes.values())
            )
        )

    def normalize_key(
        self,
        key: str
    ) -> str | None:
        match = self.long_key_pattern.fullmatch(key)
        if match is not None:
            # translate long key to short key
            d = match.groupdict()
            for k, v in d.items():
                if k.endswith("_") or v is None:
                    continue

                key = self.reverse_prefixes[v]
                key = key + d[k + "_"]
            return key
        
        match = self.short_key_pattern.fullmatch(key)
        if match is not None:
            return key
        
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


def load_kg_index(
    index_path: str,
    redirects_path: str | None = None,
    prefixes_path: str | None = None,
    progress: bool = False
) -> KgIndex:
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
            short = split[0].strip()
            obj_names = [n.strip() for n in split[1:]]
            assert short not in index, \
                f"duplicate id {short}"
            index[short] = obj_names

    redirect = {}
    if redirects_path is not None:
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
                short = split[0].strip()
                for redir in split[1:]:
                    redir = redir.strip()
                    assert redir not in redirect, \
                        f"duplicate redirect {redir}, should not happen"
                    redirect[redir] = short

    prefixes = {}
    if prefixes_path is not None:
        prefixes = load_prefixes(prefixes_path)

    return KgIndex(index, redirect, prefixes)


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
    parser: grammar.LR1Parser
) -> str:
    parse = parser.parse(sparql, skip_empty=True, collapse_single=False)

    # some simple rules for pretty printing:
    # 1. new lines after prologue (PrologueDecl) and triple blocks (TriplesBlock)
    # 2. new lines after { and before }
    # 3. increase indent after { and decrease before }

    indent = 0
    
    def _pretty(parse: dict) -> str:
        nonlocal indent
        s = ""

        if parse["name"] == "}":
            indent -= 2
            s += "\n"

        if "value" in parse:
            s += parse["value"]
        else:
            s += " ".join(_pretty(p) for p in parse["children"])

        if parse["name"] == "PrologueDecl":
            s += "\n"
        if parse["name"] == "TriplesBlock":
            s += "\n"
        if parse["name"] == "{":
            indent += 2
            s += "\n"

        return s

    return _pretty(parse)


def _find_with_name(
    parse: dict,
    name: str
) -> dict | None:
    if parse["name"] == name:
        return parse
    for child in parse.get("children", []):
        t = _find_with_name(child, name)
        if t is not None:
            return t
    return None


def _find_all_with_name(
    parse: dict,
    name: str
) -> Iterator[dict]:
    if parse["name"] == name:
        yield parse
        return
    for child in parse.get("children", []):
        yield from _find_all_with_name(child, name)


def ask_to_select(
    sparql: str,
    parser: grammar.LR1Parser
) -> str:
    parse = parser.parse(sparql, skip_empty=False, collapse_single=True)

    sub_parse = _find_with_name(parse, "QueryType")
    if sub_parse is None:
        return _parse_to_string(parse)

    query = sub_parse["children"][0]
    if query["name"] != "AskQuery":
        return _parse_to_string(parse)

    # we have an ask query
    # find the first var that is not in a subselect
    var_parse = _find_with_name(sub_parse, "Var")
    if var_parse is not None:
        # ask query has a var, convert to select
        query["name"] = "SelectQuery"
        # replace ASK terminal with SelectClause
        query["children"][0] = {
            'name': 'SelectClause',
            'children': [
                {
                    'name': 'SELECT',
                    'value': 'SELECT',
                },
                {
                    'name': '*',
                    'value': '*',
                }
            ],
        }
        return parse

    raise NotImplementedError


_CLEAN_PATTERN = re.compile(r"\s+", flags=re.MULTILINE)

def clean(s: str) -> str:
    return _CLEAN_PATTERN.sub(" ", s).strip()

def preprocess_natural_language_query(
    query: str,
    kgs: list[str],
    information: str | None,
    examples: list[tuple[str, str]] | None
) -> str:
    if examples is None or len(examples) == 0:
        example_list = ""
    else:
        example_list = "\n" + "\n\n".join(
            f"{i+1}. Example:\n{clean(query)}\n{clean(sparql)}"
            for i, (query, sparql) in enumerate(examples)
        ) + "\n"
    kg_list = "\n".join(kgs)
    return f"""\
Task:
SPARQL query generation over the specified knowledge graphs given a natural \
language query, optional additional information / guidance, and optional examples \
of query and SPARQL pairs.

Knowledge graphs:
{kg_list}

Query:
{query}

Additional information / guidance:
{information}
{example_list}
SPARQL:
"""


_KG_PATTERN = re.compile("^<kg(?:e|p) kg='(\\w+)'>")


def postprocess_sparql_query(
    sparql: str,
    parser: grammar.LR1Parser,
    entity_indices: dict[str, ContIndex],
    property_indices: dict[str, ContIndex],
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
        if kg not in entity_indices:
            continue
        index = entity_indices[kg]
        value = index.get_value(val[kg_match.end():].encode("utf8"))
        if value is None:
            continue
        entity["value"] = value

    for prop in _find_all_with_name(parse, "KGP"):
        val = prop["value"]
        kg_match = _KG_PATTERN.search(val)
        assert kg_match is not None
        kg = kg_match.group(1)
        if kg not in property_indices:
            continue
        index = property_indices[kg]
        value = index.get_value(val[kg_match.end():].encode("utf8"))
        if value is None:
            continue
        prop["value"] = value

    sparql = _parse_to_string(parse)
    sparql = fix_prefixes(sparql, parser, prefixes)
    if pretty:
        sparql = prettify(sparql, parser)
    return sparql


class SelectRecord:
    def __init__(
        self,
        value: str | None,
        data_type: str | None,
        label: str | None = None
    ):
        self.value = value
        self.data_type = data_type
        self.label = label

    def __repr__(self) -> str:
        if self.data_type is None:
            return ""
        elif self.data_type == "uri":
            assert self.value is not None
            last = self.value.split("/")[-1]
            if self.label is not None:
                return f"{self.label} ({last})"
            return last
        else:
            return self.label or self.value or ""


AskResult = bool


class SelectResult:
    def __init__(
        self,
        vars: list[str],
        results: list[dict[str, SelectRecord]]
    ):
        self.vars = vars
        self.results = results

    def __len__(self) -> int:
        return len(self.results)

    def __repr__(self) -> str:
        return f"SPARQLResult({self.vars}, {self.results})"


def query_qlever(
    sparql_query: str,
    parser: grammar.LR1Parser,
    kg: str,
    qlever_endpoint: str | None
) -> SelectResult | AskResult:
    parse = parser.parse(sparql_query, skip_empty=False, collapse_single=True)
    query_type = _find_with_name(parse, "QueryType", set())["children"][0]["name"]
    if query_type == "AskQuery":
        sparql_query = _parse_to_string(_ask_to_select(parse))
    elif query_type != "SelectQuery":
        raise ValueError(f"unsupported query type {query_type}")

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

    if query_type == "AskQuery":
        return AskResult(len(json["results"]["bindings"]) > 0)

    vars = json["head"]["vars"]
    results = []
    for binding in json["results"]["bindings"]:
        result = {}
        for var in vars:
            if binding is None or var not in binding:
                result[var] = SelectRecord(None, None)
                continue
            value = binding[var]
            result[var] = SelectRecord(
                value["value"],
                value["type"]
            )
        results.append(result)
    return SelectResult(vars, results)


def format_qlever_result(
    result: SelectResult | AskResult,
    max_column_width: int = 80,
) -> str:
    if isinstance(result, AskResult):
        return "yes" if result else "no"

    if len(result) == 0:
        return "no results"

    if len(result.vars) == 0:
        return "no bindings"

    data = []
    for record in result.results:
        data.append([
            str(record[var]) if var in record else "-"
            for var in result.vars
        ])

    return generate_table(
        headers=[result.vars],
        data=data,
        alignments=["left"] * len(result.vars),
        max_column_width=max_column_width,
    )


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
                r[var].value or "" if var in r else ""
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


def fix_prefixes(
    sparql: str,
    parser: grammar.LR1Parser,
    prefixes: dict[str, str]
) -> str:
    """
    Clean the prefixes in the SPARQL query.

    >>> parser = grammar.LR1Parser(*_load_sparql_grammar([]))
    >>> prefixes = {"test:": "http://test.com/"}
    >>> s = "PREFIX bla: <unrelated> SELECT ?x WHERE { ?x <http://test.com/prop> ?y }"
    >>> fix_prefixes(s, parser, prefixes)
    'SELECT ?x WHERE { ?x <http://test.com/prop> ?y }'
    >>> s = "SELECT ?x WHERE { ?x test:prop ?y }"
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

    def _replace_objs(parse: dict) -> tuple[dict, int, int]:
        total = 0
        replaced = 0
        for obj in _find_all_with_name(parse, "iri"):
            child = obj["children"][0]
            if child["name"] == "PrefixedName":
                val = child["children"][0]["value"]
            elif child["name"] == "IRIREF":
                val = child["value"]
            else:
                continue

            rep = _replace_obj(val, entity_indices, entity_replacements, True)
            # if rep is unchanged, this means that the val is not supported
            # by any of the entity indices
            if rep == val:
                rep = _replace_obj(val, property_indices, property_replacements, False)

            total += 1
            if rep is not None:
                obj["value"] = rep

        return parse, replaced, total

    parse, replaced, total = _replace_objs(parse)
    incomplete = replaced < total
    sparqls = [_parse_to_string(parse)]

    done = replacement == "only_first"
    while not done:
        parse, replaced, total = _replace_objs(copy.deepcopy(org_parse))
        done = replaced < total or total == 0
        if not done:
            sparqls.append(_parse_to_string(parse))

    return sparqls, incomplete
