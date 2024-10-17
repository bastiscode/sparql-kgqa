from collections import Counter
from urllib.parse import quote_plus
import time
import requests
import tempfile
import os
import random
import logging
import re
import uuid
from importlib import resources
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, Type, TypeVar, Any

from text_utils import grammar
from search_index import PrefixIndex, QGramIndex
from search_index.index import SearchIndex
from search_index.mapping import Mapping as SearchIndexMapping
from text_utils.api.table import generate_table

from sparql_kgqa.api.utils import Chat

LOGGER = logging.getLogger(__name__)
CLEAN_PATTERN = re.compile(r"\s+", flags=re.MULTILINE)

QLEVER_API = "https://qlever.cs.uni-freiburg.de/api"
QLEVER_URLS = {
    "wikidata": f"{QLEVER_API}/wikidata",
    "dbpedia": f"{QLEVER_API}/dbpedia",
    "freebase": f"{QLEVER_API}/freebase",
    "dblp": f"{QLEVER_API}/dblp",
    "orkg": f"{QLEVER_API}/orkg",
}

OBJ_TYPES = ["entity", "property", "other", "literal"]
SEARCH_TOKEN = "<|search|>"

SelectResult = list[list[str]]
AskResult = bool


def clean(s: str) -> str:
    return CLEAN_PATTERN.sub(" ", s).strip()


def get_index_dir() -> str | None:
    return os.getenv("SEARCH_INDEX_DIR", None)


def load_sparql_grammar() -> tuple[str, str]:
    sparql_grammar = resources.read_text(
        "sparql_kgqa.sparql.grammar",
        "sparql.y"
    )
    sparql_lexer = resources.read_text(
        "sparql_kgqa.sparql.grammar",
        "sparql2.l"
    )
    return sparql_grammar, sparql_lexer


def load_iri_and_literal_grammar() -> tuple[str, str]:
    iri_and_literal_grammar = resources.read_text(
        "sparql_kgqa.sparql.grammar",
        "iri_literal.y"
    )
    iri_and_literal_lexer = resources.read_text(
        "sparql_kgqa.sparql.grammar",
        "iri_literal.l"
    )
    return iri_and_literal_grammar, iri_and_literal_lexer


def is_prefix_of_iri(prefix: str, iri: str) -> bool:
    # find / necessary becuase some prefixes are prefixes of each other,
    # e.g. <http://www.wikidata.org/entity/
    # and  <http://www.wikidata.org/entity/statement/
    return iri.startswith(prefix) and iri.find("/", len(prefix)) == -1


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


def ask_to_select(
    sparql: str,
    parser: grammar.LR1Parser,
    var: str | None = None,
    distinct: bool = False
) -> str | None:
    parse = parser.parse(sparql)

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
        sel_clause = [
            {
                'name': 'SELECT',
                'value': 'SELECT',
            },
            {
                'name': 'DISTINCT',
                'value': 'DISTINCT' * distinct
            },
            {
                'name': 'Var',
                'value': var
            }
        ]
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
                'name': 'DISTINCT',
                'value': 'DISTINCT' * distinct
            },
            {
                'name': 'Var',
                'value': var
            }
        ],
    }
    return parse_to_string(parse)


class Alternative:
    def __init__(
        self,
        identifier: str,
        short_identifier: str | None = None,
        label: str | None = None,
        variants: dict[str, str] | None = None,
        aliases: list[str] | None = None,
        infos: list[str] | None = None
    ) -> None:
        self.identifier = identifier
        self.short_identifier = short_identifier
        self.label = label
        self.aliases = aliases
        self.variants = variants
        self.infos = infos

    def __repr__(self) -> str:
        return f"Alternative({self.label}, {self.get_identifier()}, " \
            f"{self.variants})"

    @staticmethod
    def _clip(s: str | None, max_len: int = 128) -> str:
        if s is None:
            return ""

        return s[:max_len] + "..." if len(s) > max_len else s

    def get_identifier(self) -> str:
        return self.short_identifier or self.identifier

    def get_label(self) -> str:
        if self.label:
            return self._clip(self.label)

        return self.get_identifier()

    def has_variants(self) -> bool:
        return bool(self.variants)

    def get_sparql_label(self, variant: str | None = None) -> str:
        identifier = self.get_variant_or_identifier(variant)
        assert identifier is not None, \
            "failed to get sparql label due to invalid variant"
        if not self.label:
            return identifier

        return f"{self.get_label()} ({identifier})"

    def get_variant_or_identifier(
        self,
        variant: str | None
    ) -> str | None:
        if not variant:
            return self.get_identifier()
        elif not self.variants:
            return None
        else:
            return self.variants.get(variant, None)

    def get_variant_by_name(
        self,
        name: str
    ) -> str | None:
        if not self.variants:
            return None

        return next((
            item[0] for item in
            filter(
                lambda item: item[1] == name,
                self.variants.items()
            )),
            None
        )

    def get_string(
        self,
        max_aliases: int = 5,
        add_infos: bool = True
    ) -> str:
        s = self.get_label()

        desc = []
        if self.variants:
            desc.append("/".join(sorted(self.variants.values())))
        elif self.label:
            desc.append(self.get_identifier())

        if add_infos and self.infos:
            desc.extend(self._clip(info) for info in self.infos)

        if desc:
            s += " (" + ", ".join(desc) + ")"

        if add_infos and max_aliases and self.aliases:
            aliases = random.sample(
                self.aliases,
                min(len(self.aliases), max_aliases or len(self.aliases))
            )
            s += ", also known as " + ", ".join(self._clip(a) for a in aliases)

        return s

    def get_regex(self) -> str:
        r = re.escape(self.get_label())
        if self.variants:
            r += re.escape(" (") + "(?:" + "|".join(map(
                re.escape,
                self.variants.values()
            )) + ")" + re.escape(")")
        return r


class NoneAlternative(Alternative):
    def __init__(self, obj_type: str) -> None:
        super().__init__("none", "none")
        self.obj_type = obj_type

    def get_string(
        self,
        max_aliases: int | None = 5,
        add_infos: bool = True
    ) -> str:
        return f"{self.label} (if no other {self.obj_type} fits well enough)"

    def get_regex(self, variants: list[str] | None = None) -> str:
        return re.escape("none")


WIKIDATA_PROPERTY_VARIANTS = {
    "<http://www.wikidata.org/prop/direct-normalized/": "wdtn",
    "<http://www.wikidata.org/prop/direct/": "wdt",
    "<http://www.wikidata.org/prop/": "p",
    "<http://www.wikidata.org/prop/qualifier/": "pq",
    "<http://www.wikidata.org/prop/qualifier/value-normalized/": "pqn",
    "<http://www.wikidata.org/prop/qualifier/value/": "pqv",
    "<http://www.wikidata.org/prop/reference/": "pr",
    "<http://www.wikidata.org/prop/reference/value-normalized/": "prn",
    "<http://www.wikidata.org/prop/reference/value/": "prv",
    "<http://www.wikidata.org/prop/statement/": "ps",
    "<http://www.wikidata.org/prop/statement/value-normalized/": "psn",
    "<http://www.wikidata.org/prop/statement/value/": "psv",
}


class Mapping:
    IDENTIFIER_COLUMN = 3

    def __init__(self) -> None:
        self.map: SearchIndexMapping | None = None

    @classmethod
    def load(cls, index: SearchIndex, path: str) -> "Mapping":
        map = SearchIndexMapping(index, cls.IDENTIFIER_COLUMN, path)
        mapping = cls()
        mapping.map = map
        return mapping

    def __getitem__(self, key: str) -> int:
        assert self.map is not None, "mapping not loaded"
        item = self.map.get(key)
        assert item is not None, f"key '{key}' not in mapping"
        return item

    def normalize(self, iri: str) -> tuple[str, str | None] | None:
        return iri, None

    def denormalize(self, key: str, variant: str | None) -> str | None:
        return key

    def default_variants(self) -> set[str]:
        return set()

    def __contains__(self, key: str) -> bool:
        assert self.map is not None, "mapping not loaded"
        return self.map.get(key) is not None


class WikidataPropertyMapping(Mapping):
    NORM_PREFIX = "<http://www.wikidata.org/entity/"

    def __init__(self) -> None:
        super().__init__()
        self.inverse_variants = {
            var: pfx
            for pfx, var in WIKIDATA_PROPERTY_VARIANTS.items()
        }

    @staticmethod
    def _longest_prefix(key: str) -> str | None:
        longest = None
        for prefix in WIKIDATA_PROPERTY_VARIANTS:
            if not is_prefix_of_iri(prefix, key):
                continue
            if longest is None or len(prefix) > len(longest):
                longest = prefix
        return longest

    def normalize(self, iri: str) -> tuple[str, str | None] | None:
        longest = self._longest_prefix(iri)
        if longest is None:
            return None
        key = self.NORM_PREFIX + iri[len(longest):]
        return key, WIKIDATA_PROPERTY_VARIANTS[longest]

    def denormalize(self, key: str, variant: str | None) -> str | None:
        if variant is None:
            return key
        elif variant not in self.inverse_variants:
            return None
        elif not is_prefix_of_iri(self.NORM_PREFIX, key):
            return None
        pfx = self.inverse_variants[variant]
        return pfx + key[len(self.NORM_PREFIX):]

    def default_variants(self) -> set[str]:
        return {"wdt", "p"}


class KgManager:
    entity_mapping_cls: Type[Mapping] = Mapping
    property_mapping_cls: Type[Mapping] = Mapping
    prefixes = {
        "bd": "<http://www.bigdata.com/rdf#",
        "cc": "<http://creativecommons.org/ns#",
        "dct": "<http://purl.org/dc/terms/",
        "geo": "<http://www.opengis.net/ont/geosparql#",
        "hint": "<http://www.bigdata.com/queryHints#",
        "ontolex": "<http://www.w3.org/ns/lemon/ontolex#",
        "owl": "<http://www.w3.org/2002/07/owl#",
        "prov": "<http://www.w3.org/ns/prov#",
        "rdf": "<http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "<http://www.w3.org/2000/01/rdf-schema#",
        "schema": "<http://schema.org/",
        "skos": "<http://www.w3.org/2004/02/skos/core#",
        "xsd": "<http://www.w3.org/2001/XMLSchema#",
        "wikibase": "<http://wikiba.se/ontology#",
    }
    custom_prefixes: dict[str, str] = {}
    kg: str = "kg"

    def __init__(
        self,
        entity_index: SearchIndex,
        property_index: SearchIndex,
        entity_mapping: Mapping,
        property_mapping: Mapping,
    ):
        assert entity_index.get_type() == property_index.get_type(), \
            "entity and property index types do not match"
        self.entity_index = entity_index
        self.property_index = property_index
        self.entity_mapping = entity_mapping
        self.property_mapping = property_mapping
        self.parser = grammar.LR1Parser(*load_sparql_grammar())
        self.iri_literal_parser = grammar.LR1Parser(
            *load_iri_and_literal_grammar()
        )
        self.search_pattern = re.compile(re.escape(SEARCH_TOKEN))

    def get_constraint(
        self,
        continuations: list[bytes],
        exact: bool
    ) -> grammar.LR1Constraint:
        return grammar.LR1Constraint(
            *load_sparql_grammar(),
            continuations,
            exact
        )

    def prettify(
        self,
        sparql: str,
        indent: int = 2,
        is_prefix: bool = False
    ) -> str:
        try:
            if is_prefix:
                parse, rest = self.parser.prefix_parse(
                    (sparql + " ").encode(),
                    skip_empty=True,
                    collapse_single=False
                )
                rest_str = bytes(rest[:-1]).decode(errors="replace")
            else:
                parse = self.parser.parse(
                    sparql,
                    skip_empty=True,
                    collapse_single=False
                )
                rest_str = ""
        except Exception as e:
            LOGGER.debug(f"prettifying sparql '{sparql}' failed: {e}")
            return sparql

        # some simple rules for prettifing:
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

    def autocomplete_prefix(
        self,
        prefix: str,
        limit: int | None = None
    ) -> tuple[str, str] | None:
        """
        Autocomplete the SPARQL prefix by running
        it against the SPARQL grammar parser.
        Assumes the prefix is somewhere in a triple block.
        Returns the full SPARQL query and the current position
        in the query triple block (subject, property, object).
        """
        # autocomplete by adding 1 to 3 variables to the query,
        # completing and then parsing it to find the current position
        # in the query triple block
        parse, rest = self.parser.prefix_parse(prefix.encode())
        rest_str = bytes(rest).decode(errors="replace")

        # build bracket stack to fix brackets later
        bracket_stack = []
        for item in find_all(
            parse,
            {"{", "}", "(", ")"},
        ):
            if item["name"] in ["{", "("]:
                bracket_stack.append(item["name"])
                continue

            assert bracket_stack and bracket_stack[-1] == item["name"], \
                "bracket mismatch"
            bracket_stack.pop()

        if rest_str in ["{", "("]:
            bracket_stack.append(rest_str)

        def fix_latest_subselect(parse: dict, var: str):
            subsels = list(find_all(parse, "SubSelect"))
            if not subsels:
                return

            subsel = subsels[-1]
            selclause = find(subsel, "SelectClause")
            if selclause is None or len(selclause["children"]) != 3:
                return
            selclause["children"][-1] = {"name": "Var", "value": f"?{var}"}
            fix_latest_subselect(subsel, var)

        positions = ["subject", "property", "object"]
        query_and_position = None
        for num_vars in range(len(positions), 0, -1):
            position = positions[num_vars - 1]
            vars = [uuid.uuid4().hex for _ in range(num_vars)]

            full_query = prefix.strip() + " " + " ".join(f"?{v}" for v in vars)
            for b in reversed(bracket_stack):
                if b == "{":
                    full_query += " }"
                else:
                    full_query += " )"

            try:
                parse = self.parser.parse(full_query)
            except Exception:
                continue

            # replace all select vars with the last one
            select_var = vars[0]

            # replace first select or ask with select var
            query = find(parse, "QueryType")
            assert query is not None
            query = query["children"][0]
            if query["name"] == "SelectQuery":
                select_clause = query["children"][0]
                select_clause["children"][1:] = [
                    {"name": "DISTINCT", "value": "DISTINCT"},
                    {"name": "VAR1", "value": f"?{select_var}"},
                ]
            elif query["name"] == "AskQuery":
                # ask to select here
                query["name"] = "SelectQuery"
                query["children"][0] = {
                    "name": "SelectClause",
                    "children": [
                        {"name": "SELECT", "value": "SELECT"},
                        {"name": "DISTINCT", "value": "DISTINCT"},
                        {"name": "VAR1", "value": f"?{select_var}"},
                    ]
                }
            else:
                continue

            # fix subselects
            fix_latest_subselect(parse, select_var)

            final_query = parse_to_string(parse)
            # add limit clause
            if limit is not None:
                final_query += f" LIMIT {limit}"

            query_and_position = (final_query, position)
            break

        return query_and_position

    def execute_sparql(
        self,
        sparql: str,
        endpoint: str | None = None,
        timeout: float | tuple[float, float] | None = None,
        max_retries: int = 1
    ) -> SelectResult | AskResult:
        # ask_to_select returns None if sparql is not an ask query
        select_sparql = ask_to_select(sparql, self.parser)
        sparql = select_sparql or sparql

        sparql = self.fix_prefixes(sparql)

        if endpoint is None:
            assert self.kg in QLEVER_URLS, \
                f"no QLever endpoint for knowledge graph {self.kg}"
            endpoint = QLEVER_URLS[self.kg]

        response = None
        for _ in range(max(1, max_retries)):
            response = requests.post(
                endpoint,
                headers={
                    "Content-type": "application/sparql-query",
                    "Accept": "text/tab-separated-values"
                },
                data=sparql,
                timeout=timeout
            )
            if response.status_code == 200:
                break

        assert response is not None, "cannot happen"

        if response.status_code != 200:
            exception = response.json().get("exception", "")
            raise RuntimeError(exception)

        result = response.text.splitlines()
        if select_sparql is not None:
            # > 1 because of header
            return AskResult(len(result) > 1)
        else:
            return [row.split("\t") for row in result]

    def format_sparql_result(
        self,
        result: SelectResult | AskResult,
        show_top_bottom_rows: int = 10,
        show_columns: int = 5,
    ) -> str:
        # run sparql against endpoint, format result as string
        if isinstance(result, AskResult):
            return f"Ask query returned {result}"

        num_rows = len(result) - 1
        num_columns = len(result[0])
        if num_rows == 0:
            return "Select query returned an empty result"

        def format_row(row: list[str]) -> list[str]:
            new_row = []
            for c in range(min(len(row), show_columns)):
                val = row[c]
                processed = self.process_iri_or_literal(val)
                if processed is None:
                    new_row.append(val)
                    continue

                typ, formatted, _ = processed
                if typ == "literal":
                    new_row.append(formatted)
                    continue

                # for iri check whether it is in one of the mappings
                norm = self.entity_mapping.normalize(val)
                map = self.entity_mapping
                index = self.entity_index
                if norm is None or norm[0] not in map:
                    norm = self.property_mapping.normalize(val)
                    map = self.property_mapping
                    index = self.property_index

                if norm is not None and norm[0] in map:
                    name = index.get_name(map[norm[0]])
                    formatted = f"{name} ({formatted})"

                new_row.append(formatted)

            return new_row

        # generate a nicely formatted table
        header = [format_row(result[0])]
        top_start = 1
        top_end = min(show_top_bottom_rows + 1, num_rows + 1)
        data = [
            format_row(result[r])
            for r in range(top_start, top_end)
        ]

        bottom_start = num_rows + 1 - show_top_bottom_rows
        if bottom_start > top_end:
            data.append(format_row(["..."] * num_columns))

        bottom_start = max(bottom_start, top_end)
        data.extend(
            format_row(result[r])
            for r in range(bottom_start, num_rows + 1)
        )

        table = generate_table(data, header)

        return f"""\
Got {num_rows:,} row{'s' * (num_rows != 1)} for \
{num_columns:,} variable{'s' * (num_columns != 1)}, \
showing the first {show_top_bottom_rows} and last {show_top_bottom_rows} rows \
for the first {show_columns} variables at maximum below:
{table}
"""

    def get_judgement_prompt_and_regex(
        self,
        question: str,
        sparql: str,
        natural_sparql: str,
        endpoint: str | None = None,
        timeout: float | None = None,
        max_retries: int = 1,
        max_rows: int = 10,
        max_columns: int = 5,
    ) -> tuple[str, str]:
        try:
            result = self.execute_sparql(
                sparql,
                endpoint,
                timeout,
                max_retries
            )
            formatted = self.format_sparql_result(
                result,
                max_rows,
                max_columns
            )
        except Exception as e:
            formatted = f"SPARQL execution failed:\n{e}"

        prompt = f"""\
Given a question and a SPARQL query together with its execution result, \
judge whether the SPARQL query makes sense for answering the question. \
Provide a short, high level explanation of at most 128 characters and \
a final yes or no answer.

Question:
{question}

SPARQL query over {self.kg}:
{natural_sparql}

Result:
{formatted}
"""
        regex = """\
Explanation: [\\w ]{1, 128}

Answer: (?:yes|no)"""
        return prompt, regex

    def parse_judgement(
        self,
        judgement: str
    ) -> tuple[str, bool]:
        exp = "Explanation: "
        exp_start = judgement.find(exp)
        if exp_start == -1:
            return "", True

        exp_start += len(exp)
        exp_end = judgement.find("\n", exp_start)
        if exp_end == -1:
            return "", True

        explanation = judgement[exp_start:exp_end].strip()
        ans = "Answer: "
        ans_start = judgement.find(ans, exp_end)
        if ans_start == -1:
            return explanation, True

        ans_start += len(ans)
        answer = judgement[ans_start:].strip()
        return explanation, answer == "yes"

    def find_longest_prefix(
        self,
        iri: str,
        prefixes: dict[str, str] | None = None
    ) -> tuple[str, str] | None:
        return next(
            iter(sorted(
                filter(
                    lambda pair: is_prefix_of_iri(pair[1], iri),
                    (
                        prefixes
                        or (self.prefixes | self.custom_prefixes)
                    ).items()
                ),
                key=lambda pair: len(pair[1]),
                reverse=True
            )),
            None
        )

    def process_iri_or_literal(
        self,
        data: str,
        prefixes: dict[str, str] | None = None
    ) -> tuple[str, str, str | None] | None:
        try:
            parse = self.iri_literal_parser.parse(
                data,
                skip_empty=True,
                collapse_single=True
            )
        except Exception:
            return None

        match parse["name"]:
            case "IRIREF":
                formatted = self.format_iri(data, prefixes, safe=False)
                return "iri", formatted, None

            case "PNAME_LN":
                formatted = self.format_iri(data, prefixes, safe=False)
                return "iri", formatted, None

            case lit if lit.startswith("STRING"):
                return "literal", self.format_string_literal(data), None

            case "RDFLiteral":
                if len(parse["children"]) == 2:
                    # langtag
                    s, langtag = parse["children"]
                    if not langtag["value"].startswith("@en"):
                        return None

                    return (
                        "literal",
                        self.format_string_literal(s["value"]),
                        langtag["value"]
                    )

                elif len(parse["children"]) == 3:
                    # datatype
                    s, _, datatype = parse["children"]
                    return (
                        "literal",
                        self.format_string_literal(s["value"]),
                        self.format_iri(datatype["value"])
                    )

            case "INTEGER" | "DECIMAL" | "DOUBLE" | "true" | "false":
                return "literal", data, None

    def format_iri(
        self,
        iri: str,
        prefixes: dict[str, str] | None = None,
        safe: bool = False
    ) -> str:
        longest = self.find_longest_prefix(iri, prefixes)
        if longest is None:
            return iri

        short, long = longest
        val = iri[len(long):-1]

        # check if no bad characters are in the short form
        # by url encoding it and checking if it is still the same
        if not safe or quote_plus(val) == val:
            return short + ":" + val
        else:
            return iri

    def format_string_literal(self, literal: str) -> str:
        if literal.startswith("'"):
            return literal.strip("'")
        else:
            return literal.strip('"')

    def denormalize_selection(
        self,
        obj_type: str,
        identifier: str,
        variant: str | None
    ) -> str | None:
        if obj_type == "entity":
            return self.entity_mapping.denormalize(
                identifier,
                variant
            )
        elif obj_type == "property":
            return self.property_mapping.denormalize(
                identifier,
                variant
            )
        else:
            return identifier

    def fix_prefixes(
        self,
        sparql: str,
        is_prefix: bool = False,
        remove_known: bool = False
    ) -> str:
        if is_prefix:
            parse, rest = self.parser.prefix_parse(
                (sparql + " ").encode(),
                skip_empty=False,
                collapse_single=True
            )
            rest_str = bytes(rest[:-1]).decode(errors="replace")
        else:
            parse = self.parser.parse(
                sparql,
                skip_empty=False,
                collapse_single=True
            )
            rest_str = ""

        prefixes = self.prefixes | self.custom_prefixes
        reverse_prefixes = {
            long: short
            for short, long in prefixes.items()
        }

        prologue = find(parse, "Prologue")
        assert prologue is not None

        base_decls = list(find_all(prologue, "BaseDecl"))

        exist = {}
        for prefix_decl in find_all(prologue, "PrefixDecl"):
            assert len(prefix_decl["children"]) == 3
            first = prefix_decl["children"][1]["value"]
            second = prefix_decl["children"][2]["value"]

            short = first.split(":", 1)[0]
            long = second[:-1]
            exist[short] = long

        seen = set()
        for iri in find_all(parse, "IRIREF", skip={"Prologue"}):
            formatted = self.format_iri(iri["value"], safe=True)
            if formatted == iri["value"]:
                continue

            pfx, _ = formatted.split(":", 1)
            iri["value"] = formatted
            iri["name"] = "PNAME_LN"
            seen.add(pfx)

        for pfx in find_all(
            parse,
            {"PNAME_NS", "PNAME_LN"},
            skip={"Prologue"}
        ):
            short, val = pfx["value"].split(":", 1)
            long = exist.get(short, "")

            if reverse_prefixes.get(long, short) != short:
                # replace existing short forms with our own short form
                short = reverse_prefixes[long]
                pfx["value"] = f"{short}:{val}"

            seen.add(short)

        prologue["children"] = base_decls

        for pfx in seen:
            if pfx in prefixes:
                if remove_known:
                    continue
                long = prefixes[pfx]
            elif pfx in exist:
                long = exist[pfx]
            else:
                continue

            prologue["children"].append(
                {
                    "name": "PrefixDecl",
                    "children": [
                        {"name": "PREFIX", "value": "PREFIX"},
                        {"name": "PNAME_NS", "value": f"{pfx}:"},
                        {"name": "IRIREF", "value": f"{long}>"},
                    ]
                }
            )

        return parse_to_string(parse) + rest_str

    def replace_iri(
        self,
        parse: dict[str, Any],
        with_iri: bool = True
    ) -> bool:
        if parse["name"] == "PNAME_LN":
            short = parse["value"]
            # convert to long form iri
            pfx, val = short.split(":", 1)
            if pfx not in self.custom_prefixes:
                return False

            iri = self.custom_prefixes[pfx] + val + ">"

        elif parse["name"] == "IRIREF":
            start, end = parse["byte_span"]
            if start == end:
                return False

            iri = parse["value"]

        else:
            return False

        norm = self.entity_mapping.normalize(iri)
        map = self.entity_mapping
        index = self.entity_index
        if norm is None or norm[0] not in map:
            norm = self.property_mapping.normalize(iri)
            map = self.property_mapping
            index = self.property_index

        if norm is None or norm[0] not in map:
            return True

        key, variant = norm
        label = index.get_name(map[key])

        if with_iri:
            label += f" ({self.format_iri(iri)})"
        elif variant:
            label += f" ({variant})"

        parse["name"] = "IRIREF"
        parse.pop("children", None)
        parse["value"] = f"<{label}>"
        return False

    def replace_iris(
        self,
        sparql: str,
        is_prefix: bool = False,
        with_iri: bool = True
    ) -> tuple[str, bool]:

        if is_prefix:
            parse, rest = self.parser.prefix_parse(
                (sparql + " ").encode(),
                skip_empty=True,
                collapse_single=False
            )
            rest_str = bytes(rest[:-1]).decode(errors="replace")
        else:
            parse = self.parser.parse(
                sparql,
                skip_empty=True,
                collapse_single=False
            )
            rest_str = ""

        incomplete = False

        for obj in find_all(parse, {"PNAME_LN", "IRIREF"}, skip={"Prologue"}):
            iri_incomplete = self.replace_iri(obj, with_iri)
            incomplete |= iri_incomplete

        if not incomplete:
            # remove custom prefixes if they are not used
            for pfx in find_all(
                parse,
                "PrefixDecl"
            ):
                if len(pfx["children"]) != 3:
                    continue

                short = pfx["children"][1]["value"][:-1]
                if short not in self.custom_prefixes:
                    continue

                pfx["children"] = []

        return parse_to_string(parse) + rest_str, incomplete

    def replace_entities_and_properties(self, sparql: str) -> str:
        parse = self.parser.parse(
            sparql,
            skip_empty=True,
            collapse_single=True
        )

        for obj in find_all(
            parse,
            {"IRIREF", "PNAME_LN", "PNAME_NS"},
            skip={"Prologue"}
        ):
            if "value" not in obj:
                continue

            elif obj["name"] in ["PNAME_NS", "PNAME_LN"]:
                # translate short to long
                short = obj["value"]
                pfx, val = short.split(":")
                if pfx not in self.custom_prefixes:
                    continue
                val = self.custom_prefixes[pfx] + val + ">"

            else:
                val = obj["value"]

            # try entity first
            norm = self.entity_mapping.normalize(val)
            map = self.entity_mapping
            index = self.entity_index
            if norm is None or norm[0] not in map:
                # fallback to property
                norm = self.property_mapping.normalize(val)
                map = self.property_mapping
                index = self.property_index

            # if norm is still none or key not in map, continue
            if norm is None or norm[0] not in map:
                continue

            key, variant = norm
            data = index.get_row(map[key])
            label = data.split("\t")[0]
            if variant is not None:
                label += f" ({variant})"

            obj["value"] = f"<{label}>"

        return parse_to_string(parse)

    def build_variants(
        self,
        mapping: Mapping,
        identifier: str,
        variants: set[str] | None = None
    ) -> dict[str, str]:
        denorm_variants = {}
        for var in (variants or mapping.default_variants()):
            denorm = mapping.denormalize(identifier, var)
            if denorm is None:
                continue

            denorm_variants[var] = self.format_iri(denorm)

        return denorm_variants

    def build_alternative(
        self,
        data: str,
        map: Mapping | None = None,
        variants: set[str] | None = None
    ) -> Alternative:
        label, _, syns, id, infos = data.rstrip("\r\n").split("\t")
        return Alternative(
            id,
            short_identifier=self.format_iri(id),
            label=label,
            variants=self.build_variants(
                map,
                id,
                variants
            ) if map is not None else None,
            aliases=[s for s in syns.split(";;;") if s != ""],
            infos=[i for i in infos.split(";;;") if i != ""]
        )

    def parse_autocompletions(
        self,
        result: set[str],
    ) -> tuple[
        dict[int, set[str]],
        dict[int, set[str]],
        list[tuple[str, str, str]],
        list[tuple[str, str, str]]
    ]:
        entities = {}
        properties = {}
        others = []
        literals = []
        for res in result:
            processed = self.process_iri_or_literal(res)
            if processed is None:
                continue

            typ, formatted, info = processed
            if typ == "literal":
                literals.append((res, formatted, info))
                continue

            unmatched = True
            for id_map, map in [
                (entities, self.entity_mapping),
                (properties, self.property_mapping)
            ]:
                norm = map.normalize(res)
                if norm is None:
                    continue

                iri, variant = norm
                if iri not in map:
                    continue

                id = map[iri]
                if id not in id_map:
                    id_map[id] = set()

                if variant is not None:
                    id_map[id].add(variant)

                unmatched = False

            if unmatched:
                others.append((res, formatted, info))

        # sort others by whether they are from one of our known
        # prefixes or not
        others.sort(key=lambda item: self.find_longest_prefix(item[0]) is None)
        return entities, properties, others, literals

    def get_autocompletion_result(
        self,
        prefix: str,
        max_candidates: int | None = None,
        endpoint: str | None = None,
        timeout: float | tuple[float, float] | None = None,
        max_retries: int = 1,
    ) -> tuple[set[str] | None, str] | None:
        autocompletion = self.autocomplete_prefix(
            prefix,
            max_candidates + 1
            if max_candidates is not None else None,
        )
        if autocompletion is None:
            return None

        sparql, position = autocompletion
        try:
            result = self.execute_sparql(
                sparql,
                endpoint,
                timeout,
                max_retries
            )
        except Exception as e:
            LOGGER.debug(
                f"getting autocompletion result with sparql\n{sparql}\n"
                f"failed with error:\n{e}"
            )
            return None, position

        # some checks
        if not isinstance(result, list):
            return None, position
        elif not all(len(row) == 1 for row in result):
            return None, position

        # skip header
        return {result[i][0] for i in range(1, len(result))}, position

    def get_entity_alternatives(
        self,
        id_map: dict[int, set[str]] | None = None,
        query: str | None = None,
        k: int = 10,
    ) -> list[Alternative]:
        return self.get_index_alternatives(
            self.entity_index,
            self.entity_mapping,
            id_map,
            query,
            k
        )

    def get_property_alternatives(
        self,
        id_map: dict[int, set[str]] | None = None,
        query: str | None = None,
        k: int = 10,
    ) -> list[Alternative]:
        return self.get_index_alternatives(
            self.property_index,
            self.property_mapping,
            id_map,
            query,
            k
        )

    def get_index_alternatives(
        self,
        index: SearchIndex,
        mapping: Mapping | None = None,
        id_map: dict[int, set[str]] | None = None,
        query: str | None = None,
        k: int = 10,
    ) -> list[Alternative]:
        if id_map is not None:
            index = index.sub_index_by_ids(list(id_map))

        if query is None:
            if id_map is None:
                ids = list(range(min(k, len(index))))
            else:
                ids = sorted(id_map)[:k]
        else:
            ids = [id for id, _ in index.find_matches(query)[:k]]

        return [
            self.build_alternative(
                index.get_row(id),
                mapping,
                id_map[id] if id_map is not None else None
            ) for id in ids
        ]

    def get_temporary_index_alternatives(
        self,
        data: list[tuple[str, str, str]],
        query: str | None = None,
        k: int = 10,
        index_cls: Type[SearchIndex] = PrefixIndex,
    ) -> list[Alternative]:
        with tempfile.TemporaryDirectory() as temp_dir:
            # build index and search in it
            data_file = os.path.join(temp_dir, "data.tsv")
            index_dir = os.path.join(temp_dir, "index")
            os.makedirs(index_dir, exist_ok=True)
            LOGGER.debug(
                f"building temporary index in {temp_dir} "
                f"with data at {data_file} and index in {index_dir}"
            )

            # write raw data to temp file in temp dir
            with open(data_file, "w") as f:
                f.write("label\tscore\tsynonyms\tid\tinfos\n")
                for res, formatted, info in data:
                    f.write(f"{formatted}\t0\t\t{res}\t{info}\n")

            index_cls.build(
                data_file,
                index_dir,

            )
            index = index_cls.load(data_file, index_dir)
            if query is None:
                ids = list(range(min(k, len(data))))
            else:
                ids = [id for id, _ in index.find_matches(query)[:k]]

            return [self.build_alternative(index.get_row(id)) for id in ids]

    def get_selection_alternatives(
        self,
        prefix: str,
        search: str,
        k: int,
        max_candidates: int | None = None,
        endpoint: str | None = None,
        timeout: float | tuple[float, float] | None = None,
        max_retries: int = 1,
        skip_autocomplete: bool = False,
    ) -> dict[str, list[Alternative]]:
        result = None
        try:
            if not skip_autocomplete:
                result = self.get_autocompletion_result(
                    prefix,
                    max_candidates + 1
                    if max_candidates is not None else None,
                    endpoint,
                    timeout,
                    max_retries
                )
                LOGGER.debug(
                    f"Got {len(result or set())} results "
                    f"for prefix {prefix}"
                )
        except Exception as e:
            LOGGER.debug(
                f"autocompleting prefix {prefix} failed: {e}"
            )

        if result is None:
            objects, position = None, None
        else:
            objects, position = result

        # determine which alternatives to add
        # entities can be subjects and objects
        add_entities = (position or "subject") in ["subject", "object"]
        # properties can only be properties
        add_properties = (position or "property") == "property"
        # literals can only be objects
        add_literals = (position or "object") == "object"
        # other iris can always be subjects, properties, and objects

        alternatives = {}
        if objects is None or len(objects) > (max_candidates or len(objects)):
            # objects being None means that there is no way
            # to constrain / filter the knowledge graph with the
            # current prefix, just search in the full index
            # with the search query;
            # we also do this if the number of results is greater
            # than max_results, because creating an extra index
            # for that would be too expensive
            if add_entities:
                alternatives["entity"] = self.get_entity_alternatives(
                    None,
                    search,
                    k
                )
            if add_properties:
                alternatives["property"] = self.get_property_alternatives(
                    None,
                    search,
                    k
                )
            return alternatives

        # split result into entities, properties, other iris
        # and literals
        start = time.perf_counter()
        (
            entity_map,
            property_map,
            others,
            literals
        ) = self.parse_autocompletions(objects)
        end = time.perf_counter()
        LOGGER.debug(
            f"parsing {len(objects):,} autocompletion results "
            f"took {1000 * (end - start):.2f}ms"
        )

        start = time.perf_counter()
        if add_entities:
            alternatives["entity"] = self.get_entity_alternatives(
                entity_map,
                search,
                k
            )
        if add_properties:
            alternatives["property"] = self.get_property_alternatives(
                property_map,
                search,
                k
            )
        end = time.perf_counter()
        LOGGER.debug(
            f"getting entity and property alternatives "
            f"took {1000 * (end - start):.2f}ms"
        )

        start = time.perf_counter()
        alternatives["other"] = self.get_temporary_index_alternatives(
            others,
            search,
            k
        )
        if add_literals:
            alternatives["literal"] = self.get_temporary_index_alternatives(
                literals,
                search,
                k,
            )
        end = time.perf_counter()
        LOGGER.debug(
            f"getting other and literal alternatives "
            f"took {1000 * (end - start):.2f}ms"
        )

        return alternatives

    def get_selection_prompt_and_regex(
        self,
        question: str,
        prefix: str,
        search_query: str,
        alternatives: dict[str, list[Alternative]],
        max_aliases: int = 5,
        add_infos: bool = True,
        selections: list[tuple[str, str, str | None]] | None = None,
        failures: set[tuple[str, str, str | None]] | None = None,
        add_none_alternative: bool = True
    ) -> tuple[str, str]:
        failed = []
        for obj_type, identifier, variant in failures or set():
            if obj_type not in alternatives:
                continue

            nxt = next(
                (alt for alt in enumerate(alternatives[obj_type])
                 if alt[1].identifier == identifier),
                None
            )
            if nxt is None:
                continue

            idx, alt = nxt
            offset = sum(
                len(alternatives[obj_type])
                for obj_type in OBJ_TYPES[:OBJ_TYPES.index(obj_type)]
                if obj_type in alternatives
            )
            idx = offset + idx
            failed.append(f"{idx + 1}. {alt.get_sparql_label(variant)}")

        prefix = prefix + "..."

        alt_strings = {}
        alt_regexes = {}
        alt_idx = 0
        for obj_type in OBJ_TYPES:
            if obj_type not in alternatives:
                continue

            alts = alternatives[obj_type]
            if len(alts) == 0:
                continue

            counts = Counter(
                alternative.get_identifier().lower()
                for alternative in alts
            )
            strings = []
            regexes = []
            for alternative in alts:
                alt_label = alternative.get_identifier()
                alt_idx_str = f"{alt_idx + 1}. "
                strings.append(alt_idx_str + alternative.get_string(
                    max_aliases,
                    # add info to non unique labels
                    add_infos or counts[alt_label.lower()] > 1
                ))
                regexes.append(
                    re.escape(alt_idx_str)
                    + alternative.get_regex()
                )

                alt_idx += 1

            alt_strings[obj_type] = strings
            alt_regexes[obj_type] = regexes

        alt_string = "\n\n".join(
            f"{obj_type.capitalize()} alternatives:\n" +
            "\n".join(alt_strings[obj_type])
            for obj_type in OBJ_TYPES
            if obj_type in alt_strings
        )

        alt_regex = "(?:" + "|".join(
            regex
            for obj_type in OBJ_TYPES
            if obj_type in alt_regexes
            for regex in alt_regexes[obj_type]
        )

        # none alternative
        num_alts = sum(len(alts) for alts in alternatives.values())
        if add_none_alternative or num_alts == 0:
            alt_string = "0. none (if no other alternative fits well enough)" \
                + "\n\n" * (num_alts > 0) + alt_string
            alt_regex += "|" * (num_alts > 0) + re.escape("0. none")
        alt_regex += ")"

        prompt = f"""\
Select the most fitting alternative to continue the SPARQL \
query with. The question to be answered, the current SPARQL prefix, the \
list of possible alternatives and the search query that \
returned these alternatives are given below. Avoid selecting \
alternatives that were already tried and unsuccessful.

Question:
{question.strip()}

SPARQL prefix over {self.kg}:
{prefix}
"""

        if selections:
            prompt += f"""
{self.format_selections(selections)}
"""

        prompt += f"""
Search query:
{search_query}

{alt_string}
"""

        if failed:
            failed = "\n".join(failed)
            prompt += f"""
The following alternatives were already selected and unsuccessful:
{failed}
"""

        prompt += """
Selection:
"""

        return prompt, alt_regex

    def parse_selection(
        self,
        alternatives: dict[str, list[Alternative]],
        result: str
    ) -> tuple[str, tuple[str, str, str | None]] | None:
        num, selection = result.split(". ", 1)
        idx = int(num)
        if idx == 0:
            # the none alternative was selected
            return None

        # convert selection index to offset
        idx -= 1

        offset = 0
        obj_type = OBJ_TYPES[0]
        for obj_type in OBJ_TYPES:
            obj_alts = len(alternatives.get(obj_type, []))
            if offset <= idx < offset + obj_alts:
                break
            offset += obj_alts

        alternative = alternatives[obj_type][idx - offset]

        if alternative.has_variants():
            label = alternative.get_label()
            rest = selection[len(label) + 2:-1]
            variant = alternative.get_variant_by_name(rest)
        else:
            variant = None

        name = "<" + alternative.get_sparql_label(variant) + ">"
        return name, (obj_type, alternative.identifier, variant)

    def get_search_prompt_and_regex(
        self,
        question: str,
        prefix: str,
        selections: list[tuple[str, str, str | None]] | None = None,
        failures: set[str] | None = None
    ) -> tuple[str, str]:
        prefix = prefix + "..."

        if isinstance(self.entity_index, PrefixIndex):
            index_info = "keyword prefix"
            dist_info = "number of keyword matches"
        else:
            assert isinstance(self.entity_index, QGramIndex)
            if self.entity_index.distance == "ied":
                dist = "substring"
            else:
                dist = "prefix"
            index_info = "character-level n-gram"
            dist_info = f"{dist} distance"

        prompt = f"""\
Generate a search query for the next IRI or literal to continue the SPARQL \
query with. The search query will be executed over {index_info} indices \
containing candidate IRIs and literals. It should be short and \
concise, retrieving candidates by {dist_info}. Avoid search queries that \
were already tried and unsuccesful. The question to be answered \
and the current SPARQL prefix are given below.

Question:
{question.strip()}

SPARQL prefix over {self.kg}:
{prefix}
"""
        if selections:
            prompt += f"""
{self.format_selections(selections)}
"""

        # only lowercase ascii + space, non-empty, up to 128 characters
        if failures:
            failed = "\n".join(failures)
            prompt += f"""
The following search queries were already tried and unsuccessful:
{failed}
"""

        prompt += """
Search query:
"""

        return prompt, r"[\S ]{1,128}"

    def get_sparql_prompt(self, question: str) -> str:
        return f"""\
Generate a natural language SPARQL query over {self.kg} to answer the given \
question.

Question:
{question.strip()}

SPARQL query:
"""

    def format_selections(
        self,
        selections: list[tuple[str, str, str | None]],
    ) -> str:
        entities = {}
        properties = {}

        for obj_type, ident, variant in selections:
            if obj_type == "entity" and ident in self.entity_mapping:
                id = self.entity_mapping[ident]
                if id not in entities:
                    entities[id] = set()
                if variant is not None:
                    entities[id].add(variant)

            elif obj_type == "property" and ident in self.property_mapping:
                id = self.property_mapping[ident]
                if id not in properties:
                    properties[id] = set()
                if variant is not None:
                    properties[id].add(variant)

        formatted = []
        for obj_type, id_map, index, map in [
            ("entities", entities, self.entity_index, self.entity_mapping),
            ("properties", properties, self.property_index,
             self.property_mapping)
        ]:
            if not id_map:
                continue

            formatted.append((
                obj_type,
                "\n".join(
                    self.build_alternative(
                        index.get_row(id),
                        map,
                        variants
                    ).get_string()
                    for id, variants in id_map.items()
                )
            ))

        return "\n\n".join(
            f"Using {obj_type}:\n{selection}"
            for obj_type, selection in formatted
        )

    def get_sparql_continuation_prompt(
        self,
        question: str,
        sparql_prefix: str,
        natural_prefix: str,
        selections: list[tuple[str, str, str | None]] | None = None,
        failures: set[str] | None = None,
        examples: list[tuple[str, str]] | None = None
    ) -> Chat:
        prompt = f"""\
Continue the SPARQL prefix to answer the question either \
until the end of the SPARQL query or the next {self.kg} \
knowledge graph search via {SEARCH_TOKEN}. Avoid continuations \
that were already tried and unsuccesful.

Question:
{question.strip()}

SPARQL prefix over {self.kg}:
{natural_prefix}
"""

        if selections:
            prompt += f"""
{self.format_selections(selections)}
"""

        if failures:
            failed = "\n".join(failures)
            prompt += f"""
The following continuations were already generated and unsuccessful:
{failed}
"""
        prompt += """
Continuation:
"""

        messages = []
        for q, s in examples or []:
            try:
                s = self.fix_prefixes(s, remove_known=True)
                s, _ = self.replace_iris(s, with_iri=False)
            except Exception:
                # skip invalid examples
                continue

            messages.extend([
                {
                    "role": "user",
                    "text": self.get_sparql_prompt(q)
                },
                {
                    "role": "assistant",
                    "text": s
                }
            ])

        # add actual question
        messages.extend([
            {
                "role": "user",
                "text": prompt
            },
            {
                "role": "assistant",
                "text": sparql_prefix,
                "partial": True
            }
        ])

        return messages


class DBLPManager(KgManager):
    kg = "dblp"

    def __init__(
        self,
        entity_index: SearchIndex,
        property_index: SearchIndex,
        entity_mapping: Mapping,
        property_mapping: Mapping,
    ):
        super().__init__(
            entity_index,
            property_index,
            entity_mapping,
            property_mapping,
        )
        # add dblp specific prefixes
        self.custom_prefixes.update({
            "dblp": "<https://dblp.org/rdf/schema#",
        })


class FreebaseManager(KgManager):
    kg = "freebase"

    def __init__(
        self,
        entity_index: SearchIndex,
        property_index: SearchIndex,
        entity_mapping: Mapping,
        property_mapping: Mapping,
    ):
        super().__init__(
            entity_index,
            property_index,
            entity_mapping,
            property_mapping,
        )
        # add freebase specific prefixes
        self.custom_prefixes.update({
            "fb": "<http://rdf.freebase.com/ns/",
        })


class DBPediaManager(KgManager):
    kg = "dbpedia"

    def __init__(
        self,
        entity_index: SearchIndex,
        property_index: SearchIndex,
        entity_mapping: Mapping,
        property_mapping: Mapping,
    ):
        super().__init__(
            entity_index,
            property_index,
            entity_mapping,
            property_mapping,
        )
        # add dbpedia specific prefixes
        self.custom_prefixes.update({
            "dbp": "<http://dbpedia.org/property/",
            "dbo": "<http://dbpedia.org/ontology/",
            "dbr": "<http://dbpedia.org/resource/",
        })


class ORKGManager(KgManager):
    kg = "orkg"

    def __init__(
        self,
        entity_index: SearchIndex,
        property_index: SearchIndex,
        entity_mapping: Mapping,
        property_mapping: Mapping,
    ):
        super().__init__(
            entity_index,
            property_index,
            entity_mapping,
            property_mapping,
        )
        # add dbpedia specific prefixes
        self.custom_prefixes.update({
            "orkgc": "<http://orkg.org/orkg/class/",
            "orkgp": "<http://orkg.org/orkg/predicate/",
            "orkgsh": "<http://orkg.org/orkg/shapes/",
            "orkgr": "<http://orkg.org/orkg/resource/",
        })


class WikidataManager(KgManager):
    property_mapping_cls = WikidataPropertyMapping
    kg = "wikidata"

    def __init__(
        self,
        entity_index: SearchIndex,
        property_index: SearchIndex,
        entity_mapping: Mapping,
        property_mapping: WikidataPropertyMapping,
    ):
        super().__init__(
            entity_index,
            property_index,
            entity_mapping,
            property_mapping,
        )
        # add wikidata specific prefixes
        self.custom_prefixes.update({
            "wd": "<http://www.wikidata.org/entity/",
            "wds": "<http://www.wikidata.org/entity/statement/",
            "wdref": "<http://www.wikidata.org/reference/",
            **{
                short: long
                for long, short in WIKIDATA_PROPERTY_VARIANTS.items()
            }
        })


KG_TO_MANAGER: dict[str, Type[KgManager]] = {
    "freebase": FreebaseManager,
    "dbpedia": DBPediaManager,
    "dblp": DBLPManager,
    "wikidata": WikidataManager,
    "orkg": ORKGManager,
}


def get_kg_manager(
    kg: str,
    entity_index: SearchIndex,
    property_index: SearchIndex,
    entity_mapping: Mapping,
    property_mapping: Mapping,
) -> KgManager:
    if kg not in KG_TO_MANAGER:
        raise ValueError(f"unknown knowledge graph {kg}")

    kg_manager_cls = KG_TO_MANAGER[kg]
    return kg_manager_cls(
        entity_index,
        property_index,
        entity_mapping,
        property_mapping
    )


def load_index_and_mapping(
    index_dir: str,
    index_type: str,
    mapping_cls: Type[Mapping] | None = None,
    **kwargs: Any
) -> tuple[SearchIndex, Mapping]:
    if index_type == "prefix":
        index_cls = PrefixIndex
    elif index_type == "qgram":
        index_cls = QGramIndex
    else:
        raise ValueError(f"unknown index type {index_type}")

    index = index_cls.load(
        os.path.join(index_dir, "data.tsv"),
        os.path.join(index_dir, index_type),
        **kwargs
    )

    if mapping_cls is None:
        mapping_cls = Mapping

    mapping = mapping_cls.load(
        index,
        os.path.join(index_dir, index_type, "index.mapping")
    )
    return index, mapping


def load_kg_manager(
    kg: str,
    entities: str | None = None,
    properties: str | None = None,
    index_type: str = "prefix"
) -> KgManager:
    index_dir = get_index_dir()
    if entities is None:
        assert index_dir is not None, \
            "SEARCH_INDEX_DIR environment variable not set"
        entities = os.path.join(index_dir, f"{kg}-entities")

    if properties is None:
        assert index_dir is not None, \
            "SEARCH_INDEX_DIR environment variable not set"
        properties = os.path.join(index_dir, f"{kg}-properties")

    ent_index, ent_mapping = load_index_and_mapping(
        entities,
        index_type
    )

    is_wikidata = kg == "wikidata"
    prop_index, prop_mapping = load_index_and_mapping(
        properties,
        index_type,
        WikidataPropertyMapping if is_wikidata else None
    )
    return get_kg_manager(
        kg,
        ent_index,
        prop_index,
        ent_mapping,
        prop_mapping
    )


def run_parallel(
    fn: Callable,
    inputs: Iterable,
    n: int | None = None
) -> Iterator:
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = []
        for input in inputs:
            future = executor.submit(fn, *input)
            futures.append(future)

        for future in futures:
            yield future.result()


T = TypeVar("T")


def flatten(
    iter: Iterable[Iterable[T]]
) -> Iterator[T]:
    for sub_iter in iter:
        yield from sub_iter


def enumerate_flatten(
    iter: Iterable[Iterable[T]]
) -> Iterator[tuple[int, T]]:
    for i, sub_iter in enumerate(iter):
        for item in sub_iter:
            yield i, item


def split(
    items: list[T],
    splits: list[int]
) -> list[list[T]]:
    assert sum(splits) == len(items), "splits do not match items"
    start = 0
    result = []
    for split in splits:
        result.append(items[start:start + split])
        start += split
    return result


def partition_by(
    iter: Iterable[T],
    key: Callable[[T], bool]
) -> tuple[list[T], list[T]]:
    a, b = [], []
    for item in iter:
        if key(item):
            a.append(item)
        else:
            b.append(item)
    return a, b


