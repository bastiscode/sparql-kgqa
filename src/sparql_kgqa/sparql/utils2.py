from collections import Counter
import os
import random
import logging
import re
import uuid
from importlib import resources
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Iterator, Type, TypeVar, Any

from text_utils import grammar, tokenization
from search_index import PrefixIndex, QGramIndex
from search_index.index import SearchIndex
from search_index.mapping import Mapping as SearchIndexMapping
from text_utils.api.table import generate_table

from sparql_kgqa.sparql.utils import (
    AskResult,
    find_all,
    find,
    parse_to_string,
    ask_to_select,
    prettify,
    query_qlever
)

LOGGER = logging.getLogger(__name__)
CLEAN_PATTERN = re.compile(r"\s+", flags=re.MULTILINE)
Chat = list[dict[str, str]]


def clean(s: str) -> str:
    return CLEAN_PATTERN.sub(" ", s).strip()


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


class Alternative:
    def __init__(
        self,
        label: str,
        identifier: str,
        short_identifier: str | None = None,
        variants: list[str] | None = None,
        aliases: list[str] | None = None,
        infos: list[str] | None = None
    ) -> None:
        self.label = label
        self.identifier = identifier
        self.short_identifier = short_identifier
        self.aliases = aliases
        self.variants = variants
        self.infos = infos

    def __repr__(self) -> str:
        return f"Alternative({self.label}, {self.identifier}, {self.variants})"

    @staticmethod
    def _clip(s: str, max_len: int = 128) -> str:
        return s[:max_len] + "..." if len(s) > max_len else s

    def get_string(self, max_aliases: int = 5, add_infos: bool = False) -> str:
        s = self.label
        if add_infos and self.infos:
            info_str = ", ".join(self._clip(info) for info in self.infos)
            s += f" ({info_str})"
        if max_aliases and self.aliases:
            aliases = random.sample(
                self.aliases,
                min(len(self.aliases), max_aliases)
            )
            alias_str = ", ".join(aliases)
            s += f", also known as {alias_str}"

        if self.short_identifier and self.variants:
            s += f" ({self.short_identifier}, {'|'.join(self.variants)})"
        elif self.variants:
            s += f" ({'|'.join(self.variants)})"
        elif self.short_identifier:
            s += f" ({self.short_identifier})"
        return s

    def get_regex(self) -> str:
        r = re.escape(self.label)
        if self.variants:
            r += re.escape(" (") \
                + "|".join(map(re.escape, self.variants)) \
                + re.escape(")")
        return r


class NoneAlternative(Alternative):
    def __init__(self, obj_type: str) -> None:
        super().__init__("none", "none")
        self.obj_type = obj_type

    def get_string(self, max_aliases: int = 5, add_infos: bool = True) -> str:
        return f"{self.label} (if no other {self.obj_type} fits well enough)"

    def get_regex(self) -> str:
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
    "<http://www.wikidata.org/prop/statement/value/": "psv"
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
        return set(WIKIDATA_PROPERTY_VARIANTS.values())


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

    def __init__(
        self,
        kg: str,
        entity_index: SearchIndex,
        property_index: SearchIndex,
        entity_mapping: Mapping,
        property_mapping: Mapping,
    ):
        self.kg = kg
        self.entity_index = entity_index
        self.property_index = property_index
        self.entity_mapping = entity_mapping
        self.property_mapping = property_mapping
        sparql_grammar, sparql_lexer = load_sparql_grammar()
        self.parser = grammar.LR1Parser(
            sparql_grammar,
            sparql_lexer,
        )
        literal_grammar, literal_lexer = load_iri_and_literal_grammar()
        self.iri_literal_parser = grammar.LR1Parser(
            literal_grammar,
            literal_lexer
        )
        self.search_pattern = re.compile(r"<\|search\|>")

    def get_constraint(
        self,
        continuations: list[bytes],
        exact: bool
    ) -> grammar.LR1Constraint:
        sparql_grammar, sparql_lexer = load_sparql_grammar()
        return grammar.LR1Constraint(
            sparql_grammar,
            sparql_lexer,
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
            sparql = self.fix_prefixes(
                sparql,
                is_prefix
            )
            return prettify(
                sparql,
                self.parser,
                indent,
                is_prefix
            )
        except Exception as e:
            LOGGER.debug(
                f"prettify failed for sparql '{sparql}': {e}"
            )
            return sparql

    def autocomplete_prefix(
        self,
        prefix: str,
        qlever_endpoint: str | None = None,
        limit: int | None = None,
        max_retries: int = 1
    ) -> set[str] | None:
        """
        Autocomplete the SPARQL prefix,
        run it against Qlever and return the IRIs
        and literals that can come next.
        Assumes that the prefix is a valid SPARQL prefix
        ending with <|search|>
        """
        parse, _ = self.parser.prefix_parse(
            prefix.encode(),
            skip_empty=False,
            collapse_single=True
        )

        # find one non-empty iri
        iri = next((
            p for p in find_all(
                parse,
                {"IRIREF", "PNAME_LN", "PNAME_NS"},
                skip={"Prologue", "SubSelect"}
            ) if p["value"] != ""),
            None
        )
        if iri is None:
            # means we have a prefix but with only vars,
            # which means that there are no constraints
            return None

        # determine current position in the query:
        # subject, predicate or object

        # find all triple blocks first
        triple_blocks = list(find_all(
            parse,
            "TriplesSameSubjectPath",
            skip={"SubSelect"}
        ))
        if not triple_blocks:
            # without triples the knowledge graph can not be
            # constrained
            return None

        no_iris = all(
            all(
                iri["value"] == ""
                for iri in find_all(
                    triple_block,
                    {"IRIREF", "PNAME_NS", "PNAME_LN"}
                )
            )
            for triple_block in triple_blocks
        )
        if no_iris:
            # without iris the knowledge graph can not be
            # constrained
            return None

        last_triple = triple_blocks[-1]
        # the last triple block
        assert len(last_triple["children"]) == 2
        subj, second = last_triple["children"]
        assert second["name"] == "PropertyListPathNotEmpty"

        var = uuid.uuid4().hex
        assert len(second["children"]) == 3
        prop, obj, _ = second["children"]
        if subj["name"] == "<|search|>":
            # subject can be anything
            return None

        elif prop["name"] == "<|search|>":
            # property
            prop["name"] = "VAR1"
            prop["value"] = f"?{var}"
            obj["name"] = "VAR1"
            obj_var = uuid.uuid4().hex
            obj["value"] = f"?{obj_var}"

        elif obj["name"] == "<|search|>":
            # object
            obj["name"] = "VAR1"
            obj["value"] = f"?{var}"

        else:
            assert "unexpected case"

        # fix all future brackets
        for item in find_all(
            parse,
            {"{", "}", "(", ")", "."},
        ):
            item["value"] = item["name"]

        if limit is not None:
            # find solution modifier and add limit clause
            sol_mod = find(parse, "SolutionModifier")
            assert sol_mod is not None, "could not find solution modifier"
            children = sol_mod.get("children", [])
            children.append({
                "name": "LimitClause",
                "children": [
                    {
                        'name': 'LIMIT',
                        'value': 'LIMIT'
                    },
                    {
                        'name': 'INTEGER',
                        'value': str(limit)
                    }
                ]
            })
            sol_mod["children"] = children

        prefix = parse_to_string(parse)

        select = ask_to_select(
            prefix,
            self.parser,
            var=f"?{var}",
            distinct=True
        )
        if select is not None:
            prefix = select
        else:
            # query is not an ask query, replace
            # the selected vars with our own
            parse = self.parser.parse(
                prefix,
                skip_empty=False,
                collapse_single=False
            )
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

        uris = None
        try:
            result = query_qlever(
                prefix,
                self.parser,
                self.kg,
                qlever_endpoint,
                timeout=10.0,
                max_retries=max_retries
            )
            assert isinstance(result, list)
            uris = set(result[i][0] for i in range(1, len(result)))
        except Exception as e:
            LOGGER.debug(
                "querying qlever within autocomplete_prefix "
                f"failed for prefix '{prefix}': {e}"
            )
        return uris

    def get_formatted_result(
        self,
        sparql: str,
        qlever_endpoint: str | None = None,
        max_retries: int = 1,
        max_rows: int = 10,
        max_columns: int = 5,
    ) -> str:
        # run sparql against endpoint, format result as string
        try:
            result = query_qlever(
                sparql,
                self.parser,
                self.kg,
                qlever_endpoint,
                timeout=10.0,
                max_retries=max_retries
            )
        except Exception as e:
            LOGGER.debug(
                f"querying qlever within get_formatted_result "
                f"failed for sparql '{sparql}': {e}"
            )
            return f"Query failed with exception: {e}"

        if isinstance(result, AskResult):
            return str(result)

        num_rows = len(result) - 1
        num_columns = len(result[0])
        if num_rows == 0:
            return "Empty"

        def format_uri(s: str) -> str:
            if not s.startswith("<") or not s.endswith(">"):
                return s

            # try to find in data
            norm = self.entity_mapping.normalize(s)
            map = self.entity_mapping
            index = self.entity_index
            if norm is None or norm[0] not in map:
                norm = self.property_mapping.normalize(s)
                map = self.property_mapping
                index = self.property_index

            if norm is None or norm[0] not in map:
                return s

            key, variant = norm
            label = index.get_name(map[key])
            if variant is not None:
                label += f" ({variant})"

            longest = self.find_longest_prefix(s)
            if longest is not None:
                short, long = longest
                s = short + ":" + s[len(long):-1]

            label += f" ({s})"
            return label

        table = generate_table(
            [result[0][:max_columns]],
            [
                [format_uri(v) for v in res[:max_columns]]
                for res in result[1:max_rows + 1]
            ]
        )

        return f"""\
Got {num_rows:,} row{'s' * (num_rows != 1)} for \
{num_columns:,} variable{'s' * (num_columns != 1)}, \
showing the first {max_rows} rows for the first {max_columns} variables \
at maximum below:
{table}
"""

    def get_judgement_prompt_and_regex(
        self,
        question: str,
        sparql: str,
        natural_sparql: str,
        qlever_endpoint: str | None = None,
        max_retries: int = 1,
        max_rows: int = 10,
        max_columns: int = 5,
    ) -> tuple[str, str]:
        result = self.get_formatted_result(
            sparql,
            qlever_endpoint,
            max_retries,
            max_rows,
            max_columns
        )
        prompt = f"""\
Given a question and a SPARQL query together with its execution result, \
judge whether the SPARQL query makes sense for answering the question. \
Provide a short, high level explanation of at most 16 words and \
a final yes or no answer.

Question:
{question}

SPARQL query over {self.kg}:
{natural_sparql}

Result:
{result}
"""
        regex = """\
Explanation: (?:\\w+ ){1, 15}\\w+

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

    def fix_prefixes(
        self,
        sparql: str,
        is_prefix: bool = False
    ) -> str:
        if is_prefix:
            parse, rest = self.parser.prefix_parse(
                sparql.encode(),
                skip_empty=False,
                collapse_single=True
            )
            rest_str = bytes(rest).decode(errors="replace")
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
            if first == "" or second == "":
                continue
            short = first.split(":", 1)[0]
            long = second[:-1]
            exist[short] = long

        seen = set()
        for iri in find_all(parse, "IRIREF", skip={"Prologue"}):
            if iri["value"] == "":
                continue

            val = iri["value"]
            longest = self.find_longest_prefix(val)
            if longest is None:
                continue

            short, long = longest
            iri["value"] = short + ":" + val[len(long):-1]
            seen.add(short)

        for pfx in find_all(
            parse,
            {"PNAME_NS", "PNAME_LN"},
            skip={"Prologue"}
        ):
            if pfx["value"] == "":
                continue

            short, val = pfx["value"].split(":", 1)
            long = exist.get(short, "")

            if reverse_prefixes.get(long, short) != short:
                # replace existing short forms with our own short form
                short = reverse_prefixes[long]
                pfx["value"] = f"{short}:{val}"

            seen.add(short)

        prologue["children"] = base_decls

        for pfx in seen:
            if pfx in exist:
                long = exist[pfx]
            elif pfx in prefixes:
                long = prefixes[pfx]
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

    def replace_iri(self, parse: dict[str, Any]) -> bool:
        assert parse["name"] == "iri", "obj is not an iri parse tree"
        child = parse["children"][0]
        is_in_kg = False
        if child["name"] == "PrefixedName":
            child = child["children"][0]
            start, end = child["byte_span"]
            if start == end:
                return False

            val = child["value"]
            # convert to long form
            pfx, val = val.split(":", 1)
            if pfx not in self.custom_prefixes:
                return False

            val = self.custom_prefixes[pfx] + val + ">"
            is_in_kg = True

        elif child["name"] == "IRIREF":
            start, end = child["byte_span"]
            if start == end:
                return False

            val = child["value"]
            is_in_kg = self.find_longest_prefix(
                val,
                self.custom_prefixes
            ) is not None

        else:
            return False

        if not is_in_kg:
            return False

        norm = self.entity_mapping.normalize(val)
        map = self.entity_mapping
        index = self.entity_index
        if norm is None or norm[0] not in map:
            norm = self.property_mapping.normalize(val)
            map = self.property_mapping
            index = self.property_index

        if norm is None or norm[0] not in map:
            return True

        key, variant = norm
        label = index.get_name(map[key])
        if variant is not None:
            label += f" ({variant})"

        child["name"] = "IRIREF"
        child.pop("children", None)
        child["value"] = f"<{label}>"
        return False

    def replace_iris(
        self,
        sparql: str,
        is_prefix: bool = False
    ) -> tuple[str, bool]:

        if is_prefix:
            parse, rest = self.parser.prefix_parse(
                sparql.encode(),
                skip_empty=True,
                collapse_single=False
            )
            rest_str = bytes(rest).decode(errors="replace")
        else:
            parse = self.parser.parse(
                sparql,
                skip_empty=True,
                collapse_single=False
            )
            rest_str = ""
        incomplete = False

        for obj in find_all(parse, "iri", skip={"Prologue"}):
            iri_incomplete = self.replace_iri(obj)
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

    def build_alternatives(
        self,
        data: list[tuple[str, set[str]]],
    ) -> list[Alternative]:
        alternatives: list[Alternative] = []

        for line, variants in data:
            label, _, syns, id, infos = line.rstrip("\r\n").split("\t")

            assert all(
                alt.identifier != id
                for alt in alternatives
            ), f"duplicate identifier {id} in data"
            alternative = Alternative(
                label,
                id,
                variants=sorted(variants),
                aliases=[s for s in syns.split(";;;") if s != ""],
                infos=[i for i in infos.split(";;;") if i != ""]
            )
            alternatives.append(alternative)

        return alternatives

    def get_selection_alternatives(
        self,
        prefix: str,
        search: str,
        k: int,
        max_candidates: int | None = None,
        endpoint: str | None = None,
        max_retries: int = 1,
        **kwargs: Any
    ) -> dict[str, list[Alternative]] | None:
        try:
            result = self.autocomplete_prefix(
                prefix + "<|search|>",
                endpoint,
                max_candidates + 1
                if max_candidates is not None else None,
                max_retries
            )
        except Exception as e:
            LOGGER.debug(
                f"autocomplete_prefix failed for prefix '{prefix}': "
                f"{e}"
            )
            return None

        indices = [
            ("entity", self.entity_index, self.entity_mapping),
            ("property", self.property_index, self.property_mapping)
        ]
        all_alternatives = {}

        if result is None or len(result) > (max_candidates or len(result)):
            # select result being None means that there is no way
            # to constrain / filter the knowledge graph with the
            # current prefix, just search in the full index
            # with the guess;
            # we also do this if the number of results is greater
            # than max_results, because creating an extra index
            # for that would be too expensive
            for index_type, index, map in indices:
                data: list[tuple[str, set[str]]] = []
                for id, _ in index.find_matches(search, **kwargs)[:k]:
                    data.append((index.get_row(id), map.default_variants()))
                all_alternatives[index_type] = self.build_alternatives(data)

            return all_alternatives

        seen = set()
        for index_type, index, map in indices:
            data: list[tuple[str, set[str]]] = []
            id_map = {}
            for iri in result:
                norm = map.normalize(iri)
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

                seen.add(iri)

            if k < len(result):
                # build sub index and search in it
                sub_index = index.sub_index_by_ids(list(id_map))
                for id, _ in sub_index.find_matches(search, **kwargs)[:k]:
                    data.append((sub_index.get_row(id), id_map[id]))

            else:
                # we have less than or equal k results, just get all of them
                for id, variants in id_map.items():
                    data.append((index.get_row(id), variants))

            all_alternatives[index_type] = self.build_alternatives(data)

        def format_str(parse: dict) -> str:
            if not parse["name"].startswith("STRING"):
                return parse["value"]

            if "LONG" in parse["name"]:
                return parse["value"][3:-3]
            return parse["value"][1:-1]

        def format_iri(iri: str) -> str:
            longest = self.find_longest_prefix(iri, self.prefixes)
            if longest is None:
                return iri

            short, long = longest
            return short + ":" + iri[len(long):-1]

        other_alternatives = []
        literal_alternatives = []
        for item in result - seen:
            try:
                parse = self.iri_literal_parser.parse(
                    item,
                    skip_empty=True,
                    collapse_single=True
                )
            except Exception:
                continue

            if parse["name"] == "IRIREF":
                formatted = format_iri(parse["value"])
                if formatted == item:
                    continue

                other_alternatives.append(Alternative(
                    formatted,
                    item
                ))

            elif parse["name"].startswith("STRING"):
                literal_alternatives.append(Alternative(
                    format_str(parse["value"]),
                    item
                ))

            elif parse["name"] == "RDFLiteral":
                if len(parse["children"]) == 2:
                    # langtag
                    s, langtag = parse["children"]
                    literal_alternatives.append(Alternative(
                        format_str(s),
                        item,
                        short_identifier=langtag["value"]
                    ))

                elif len(parse["children"]) == 3:
                    # datatype
                    s, _, datatype = parse["children"]

                    literal_alternatives.append(Alternative(
                        format_str(s),
                        item,
                        short_identifier=format_iri(datatype["value"])
                    ))

            elif parse["name"] in [
                "INTEGER",
                "DECIMAL",
                "DOUBLE",
                "true",
                "false"
            ]:
                literal_alternatives.append(Alternative(
                    item,
                    item
                ))

        all_alternatives["other"] = other_alternatives
        all_alternatives["literal"] = literal_alternatives
        return all_alternatives

    def get_selection_prompt_and_regex(
        self,
        question: str,
        prefix: str,
        search_query: str,
        all_alternatives: dict[str, list[Alternative]],
        max_aliases: int = 5,
        add_infos: bool = False,
        failures: set[tuple[str, str]] | None = None
    ) -> tuple[str, str]:
        prefix = prefix + "..."

        all_alts = []
        alt_idx = 0
        obj_types = ["entity", "property", "other", "literal"]
        for obj_type in obj_types:
            alternatives = all_alternatives.get(obj_type, None)
            if alternatives is None:
                continue

            counts = Counter(
                alternative.label.lower()
                for alternative in alternatives
            )
            alt_strings = []
            alt_regexes = []
            for alternative in alternatives:
                alt_idx_str = f"{alt_idx + 1}. "
                alt_strings.append(alt_idx_str + alternative.get_string(
                    max_aliases,
                    # add info to non unique labels
                    add_infos or counts[alternative.label.lower()] > 1
                ))
                r = re.escape(alt_idx_str + alternative.label)
                if alternative.variants:
                    vars = "|".join(
                        re.escape(v)
                        for v in alternative.variants
                    )
                    r += re.escape(" (") \
                        + "(?:" + vars + ")" \
                        + re.escape(")")
                alt_regexes.append(r)

            alt_idx += 1

            all_alts.append((obj_type, (alt_strings, alt_regexes)))

        num_alts = sum(len(alts) for _, (alts, _) in all_alts)

        alt_string = "\n\n".join(
            f"{obj_type.capitalize()} alternatives:\n" +
            "\n".join(alt_strings)
            for obj_type, (alt_strings, _) in all_alts
        )
        alt_regex = "(?:" + "|".join(
            regex
            for _, (_, regexes) in all_alts
            for regex in regexes
        )
        alt_string += f"\n\n{num_alts + 1}. none " \
            "(if no other alternative fits well enough)"
        alt_regex += "|" * (num_alts > 0) + re.escape(f"{num_alts + 1}. none")
        alt_regex += ")"

        failure = ""
        if failures:
            failed = []
            for obj_type, iri in failures:
                key = iri
                variant = None
                if obj_type == "entity":
                    norm = self.entity_mapping.normalize(iri)
                    assert norm is not None, \
                        f"normalizing {iri} failed"
                    key, variant = norm
                elif obj_type == "property":
                    norm = self.property_mapping.normalize(iri)
                    assert norm is not None, \
                        f"normalizing {iri} failed"
                    key, variant = norm

                offset = sum(
                    len(alts)
                    for _, (alts, _) in all_alts[:obj_types.index(obj_type)]
                )
                nxt = next(
                    (alt for alt in enumerate(all_alternatives[obj_type])
                     if alt[1].identifier == key),
                    None
                )
                assert nxt is not None, \
                    f"could not find failed alternative {key}"
                idx, alt = nxt
                fail = f"{offset + idx + 1}. {alt.label}"
                if variant is not None:
                    assert variant in (alt.variants or []), \
                        f"variant {variant} not in {alt.variants}"
                    fail += f" ({variant})"

                failed.append(fail)

            failed = "\n\n".join(failed)

            failure = f"""
The following alternatives were already tried but unsuccessful. \
If there is no other sensible alternative to try, select the none alternative:
{failed}
"""

        prompt = f"""\
Select the most fitting alternative to continue the SPARQL \
query with. The question to be answered, the current SPARQL prefix, the \
list possible alternatives and the search query that \
returned these alternatives are given below.

Question:
{question.strip()}

SPARQL prefix over {self.kg}:
{prefix}

Search query:
{search_query}

{alt_string}
{failure}
Selection:
"""
        return prompt, alt_regex

    def parse_selection(
        self,
        alternatives: list[Alternative],
        obj_type: str,
        result: str
    ) -> tuple[str, str] | None:
        num, name = result.split(".", 1)
        idx = int(num) - 1
        name = name[1:]
        if idx >= len(alternatives):
            # the none alternative was selected
            return None

        alternative = alternatives[idx]
        variant = None
        if not alternative.variants:
            # no variants to parse
            variant = None
        else:
            # parse variant
            # + 4 to account for ". " and opening " ("
            # - 1 to account for closing ")"
            variant = result[len(num) + len(alternative.label) + 4:-1]

        name = f"<{name}>"
        if obj_type == "entity":
            map = self.entity_mapping
        else:
            map = self.property_mapping

        denorm = map.denormalize(alternative.identifier, variant)
        assert denorm is not None, "denormalization failed"
        return denorm, name

    def get_search_prompt_and_regex(
        self,
        question: str,
        prefix: str,
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

        # only lowercase ascii + space, non-empty, up to 128 characters
        regex = r"[a-z0-9 ]{1,128}"
        failure = ""
        if failures:
            failed = "\n\n".join(failures)
            failure = f"""
The following search queries were already tried but unsuccessful. If there \
is no other sensible search query to try, output one of these again:
{failed}
"""
            regex += "|(?:" + "|".join(re.escape(f) for f in failures) + ")"

        prompt = f"""\
Generate a search query for the next IRI or literal to continue the SPARQL \
query with. The search query will be executed over {index_info} indices \
containing candidate IRIs and literals. It should be short and \
concise, retrieving candidates by {dist_info}. The question to be answered \
and the current SPARQL prefix are given below.

Question:
{question.strip()}

SPARQL prefix over {self.kg}:
{prefix}
{failure}
Search query:
"""
        return prompt, regex

    def get_sparql_prompt(self, question: str) -> str:
        return f"""\
Generate a natural language SPARQL query over {self.kg} to answer the given \
question.

Question:
{question.strip()}

SPARQL query:
"""

    def get_sparql_continuation_prompt(
        self,
        question: str,
        prefix: str,
        examples: list[tuple[str, str]] | None = None,
        failures: set[str] | None = None
    ) -> Chat:
        failure = ""
        if failures:
            failed = "\n\n".join(failures)
            failure = f"""
The following continuations were already tried but unsuccessful. If there \
is no other sensible continuation to try, output one of these again:
{failed}
"""
        prompt = f"""\
Continue the SPARQL prefix to answer the question either \
until the end of the SPARQL query or the next {self.kg} \
knowledge graph search via <|search|>.

Question:
{question.strip()}

SPARQL prefix over {self.kg}:
{prefix}
{failure}
Continuation:
"""

        messages = []
        for q, s in examples or []:
            try:
                s = self.fix_prefixes(self.replace_entities_and_properties(s))
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
                "text": prefix,
                "partial": True
            }
        ])

        return messages


def get_kg_manager(
    kg: str,
    entity_index: SearchIndex,
    property_index: SearchIndex,
    entity_mapping: Mapping,
    property_mapping: Mapping,
) -> KgManager:
    if kg == "freebase":
        return FreebaseManager(
            entity_index,
            property_index,
            entity_mapping,
            property_mapping
        )
    elif kg == "dbpedia":
        return DBPediaManager(
            entity_index,
            property_index,
            entity_mapping,
            property_mapping
        )
    elif kg == "dblp":
        return DBLPManager(
            entity_index,
            property_index,
            entity_mapping,
            property_mapping
        )
    elif kg == "wikidata":
        assert isinstance(property_mapping, WikidataPropertyMapping)
        return WikidataManager(
            entity_index,
            property_index,
            entity_mapping,
            property_mapping
        )
    else:
        raise ValueError(f"unknown kg {kg}")


class DBLPManager(KgManager):
    def __init__(
        self,
        entity_index: SearchIndex,
        property_index: SearchIndex,
        entity_mapping: Mapping,
        property_mapping: Mapping,
    ):
        super().__init__(
            "dblp",
            entity_index,
            property_index,
            entity_mapping,
            property_mapping,
        )
        # add dblp specific prefixes
        self.custom_prefixes.update({
            "dblp": "<https://dblp.org/rdf/schema#",
            "dblpr": "<https://dblp.org/rec/",
        })


class FreebaseManager(KgManager):
    def __init__(
        self,
        entity_index: SearchIndex,
        property_index: SearchIndex,
        entity_mapping: Mapping,
        property_mapping: Mapping,
    ):
        super().__init__(
            "freebase",
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
    def __init__(
        self,
        entity_index: SearchIndex,
        property_index: SearchIndex,
        entity_mapping: Mapping,
        property_mapping: Mapping,
    ):
        super().__init__(
            "dbpedia",
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


class WikidataManager(KgManager):
    property_mapping_cls = WikidataPropertyMapping

    def __init__(
        self,
        entity_index: SearchIndex,
        property_index: SearchIndex,
        entity_mapping: Mapping,
        property_mapping: WikidataPropertyMapping,
    ):
        super().__init__(
            "wikidata",
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

        for future in as_completed(futures):
            yield future.result()


T = TypeVar("T")


def format_obj_type(obj_type: str) -> str:
    assert obj_type in {"", "entity", "property"}
    if obj_type == "":
        return ""
    else:
        return f"<|{obj_type}|>"


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


def de_tokenize_incremental(
    tokenizer: tokenization.Tokenizer,
    initial_token_ids: list[int]
) -> Callable[[list[int]], str]:
    initial = tokenizer.de_tokenize(initial_token_ids)

    def _inc(token_ids: list[int]):
        assert all(i == t for i, t in zip(initial_token_ids, token_ids))
        dec = tokenizer.de_tokenize(token_ids)
        return dec[len(initial):]

    return _inc
