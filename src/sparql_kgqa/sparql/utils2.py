from collections import Counter
import os
import random
import logging
import re
import uuid
from importlib import resources
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Generator, Iterable, Iterator, Type, TypeVar, Any

from search_index import PrefixIndex, QGramIndex
from text_utils import grammar
from search_index.index import SearchIndex
from search_index.mapping import Mapping as SearchIndexMapping

from sparql_kgqa.sparql.utils import (
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


class Alternative:
    def __init__(
        self,
        label: str,
        identifier: str,
        variants: list[str] | None = None,
        aliases: list[str] | None = None,
        infos: list[str] | None = None
    ) -> None:
        self.label = label
        self.identifier = identifier
        self.aliases = aliases
        self.variants = variants
        self.infos = infos

    def __repr__(self) -> str:
        return f"Alternative({self.label}, {self.identifier})"

    @staticmethod
    def _clip(s: str, max_len: int = 64) -> str:
        return s[:max_len] + "..." if len(s) > max_len else s

    def get_string(self, max_aliases: int = 5, add_infos: bool = False) -> str:
        s = self.label
        if add_infos and self.infos:
            info_str = self._clip(", ".join(self.infos))
            s += f" ({info_str})"
        if self.aliases and max_aliases:
            aliases = random.sample(
                self.aliases,
                min(len(self.aliases), max_aliases)
            )
            alias_str = ", ".join(aliases)
            s += f", also known as {alias_str}"
        if self.variants:
            s += f" ({'|'.join(self.variants)})"
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
            if not key.startswith(prefix):
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
        elif not key.startswith(self.NORM_PREFIX):
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
        self.search_pattern = re.compile(r"<\|kg([ep])\|>")

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
        run it against Qlever and return the entities or
        properties that can come next.
        Assumes that the prefix is a valid SPARQL prefix
        ending with <|kge search|> or <|kgp search|>.
        """
        look_for = ["KGE", "KGP"]

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
        if subj["name"] in look_for:
            # subject can be anything
            return None

        elif prop["name"] in look_for:
            # property
            prop["name"] = "VAR1"
            prop["value"] = f"?{var}"
            obj["name"] = "VAR1"
            obj_var = uuid.uuid4().hex
            obj["value"] = f"?{obj_var}"

        elif obj["name"] in look_for:
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
            uris = set(r[0] for r in result)
        except Exception as e:
            LOGGER.debug(
                "querying qlever within autocomplete_prefix "
                f"failed for prefix '{prefix}': {e}"
            )
        return uris

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
            iri["value"] = short + ":" + val[len(long):-1]
            seen.add(short)

        for pfx in find_all(
            parse,
            {"PNAME_NS", "PNAME_LN"},
            skip={"Prologue"}
        ):
            if pfx["value"] == "":
                continue

            pfx, _ = pfx["value"].split(":", 1)
            seen.add(pfx)

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

    def replace_iri(
        self,
        parse: dict[str, Any],
        replacement: str = "label"
    ) -> bool:
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
            is_in_kg = next(filter(
                lambda pfx: val.startswith(pfx),
                self.custom_prefixes.values()
            ), None) is not None
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
        data = index.get_row(map[key])
        if replacement == "synonyms":
            syns = [
                syn for syn in
                data.split("\t")[2].split(";")
                if syn != ""
            ]
        else:
            syns = []
        syns.append(data.split("\t")[0])

        label = random.choice(syns)
        if variant is not None:
            label += f" ({variant})"

        child["name"] = "IRIREF"
        child.pop("children", None)
        child["value"] = "<iri>"
        child["value"] = f"<{label}>"
        return False

    def replace_iris(
        self,
        sparql: str,
        replacement: str = "label",
        is_prefix: bool = False
    ) -> tuple[str, bool]:
        assert replacement in [
            "label",
            "synonyms"
        ]

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
            iri_incomplete = self.replace_iri(obj, replacement)
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

    def build_alternatives_from_data(
        self,
        data: list[tuple[str, set[str]]],
    ) -> list[Alternative]:
        raise NotImplementedError

    def get_selection_alternatives(
        self,
        prefix: str,
        obj_type: str,
        search: str,
        k: int,
        max_candidates: int | None = None,
        endpoint: str | None = None,
        **kwargs: Any
    ) -> list[Alternative] | None:
        try:
            result = self.autocomplete_prefix(
                prefix + search_token_from_obj_type(obj_type),
                endpoint,
                max_candidates + 1
                if max_candidates is not None else None,
            )
        except Exception as e:
            LOGGER.debug(
                f"autocomplete_prefix failed for prefix '{prefix}': "
                f"{e}"
            )
            return None

        if obj_type == "entity":
            index = self.entity_index
            map = self.entity_mapping
        else:
            index = self.property_index
            map = self.property_mapping

        data: list[tuple[str, set[str]]] = []
        if result is None or len(result) > (max_candidates or len(result)):
            # select result being None means that there is no way
            # to constrain / filter the knowledge graph with the
            # current prefix, just search in the full index
            # with the guess;
            # we also do this if the number of results is greater
            # than max_results, because creating an extra qgram index
            # for that would be too expensive
            for id, _ in index.find_matches(search, **kwargs)[:k]:
                data.append((index.get_row(id), map.default_variants()))

        elif k < len(result):
            # build a sub index and find matches in it
            id_map = {}
            for iri in result:
                norm = map.normalize(iri)
                if norm is None:
                    continue
                iri, variant = norm
                if iri not in map:
                    continue
                id = map[iri]
                id_map[id] = set() if variant is None else {variant}

            sub_index = index.sub_index_by_ids(list(id_map))
            for id, _ in sub_index.find_matches(search, **kwargs)[:k]:
                data.append((sub_index.get_row(id), id_map[id]))

        else:
            # we have less than k result, just get all of them
            for iri in result:
                norm = map.normalize(iri)
                if norm is None:
                    continue
                iri, variant = norm
                if iri not in map:
                    continue
                data.append((
                    index.get_row(map[iri]),
                    set() if variant is None else {variant}
                ))

        return self.build_alternatives_from_data(data)

    def get_selection_prompt_and_regex(
        self,
        question: str,
        prefix: str,
        obj_type: str,
        search_query: str,
        alternatives: list[Alternative],
        add_none_alternative: bool = True,
        max_aliases: int = 5,
        add_infos: bool = False,
        failures: set[str] | None = None
    ) -> tuple[str, str]:
        assert obj_type in {"entity", "property"}
        prefix = prefix + "<...>"

        counts = Counter(
            alternative.label.lower()
            for alternative in alternatives
        )
        alt_strings = []
        alt_regexes = []
        for i, alternative in enumerate(alternatives):
            i_str = f"{i + 1}. "
            alt_strings.append(i_str + alternative.get_string(
                max_aliases,
                # add info to non unique labels
                add_infos or counts[alternative.label.lower()] > 1
            ))
            r = re.escape(i_str + alternative.label)
            if alternative.variants:
                r += re.escape(" (") \
                    + "(?:" \
                    + "|".join(re.escape(v) for v in alternative.variants) \
                    + ")" \
                    + re.escape(")")
            alt_regexes.append(r)

        if add_none_alternative:
            alt_strings.append(
                f"{len(alternatives) + 1}. none "
                f"(if no other {obj_type} fits well enough)"
            )
            alt_regexes.append(re.escape(f"{len(alternatives) + 1}. none"))

        alt_string = "\n".join(alt_strings)

        failure = ""
        if failures:
            if obj_type == "entity":
                map = self.entity_mapping
            else:
                map = self.property_mapping

            failed = []
            for f in failures:
                i, alt = next(
                    alt for alt in enumerate(alternatives)
                    if alt[1].identifier == f
                )
                fail = f"{i + 1}. {alt.label}"
                norm = map.normalize(alt.identifier)
                if norm is None or norm[1] is None:
                    failed.append(fail)
                    continue

                _, variant = norm
                fail += f" ({variant})"
                failed.append(fail)

            failed = "\n".join(failed)
            if add_none_alternative:
                stop_action = "select the none alternative"
            else:
                stop_action = "output one of these again"

            failure = f"""
The following {obj_type} alternatives were already tried but unsuccessful. \
If there is no other sensible {obj_type} alternative to try, {stop_action} \
to indicate that the search should be stopped at this point:
{failed}
"""

        prompt = f"""\
Select the most fitting {obj_type} alternative to continue the SPARQL \
query with. The question to be answered, the current SPARQL prefix, the \
list of possible {obj_type} alternatives and the index search query that \
returned these alternatives are given below.

Question:
{question.strip()}

SPARQL prefix over {self.kg}:
{prefix}

{obj_type.capitalize()} index search query:
{search_query}

{obj_type.capitalize()} alternatives:
{alt_string}
{failure}
Selection:
"""
        return prompt, "(?:" + "|".join(alt_regexes) + ")"

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
        obj_type: str,
        prefix: str,
        failures: set[str] | None = None
    ) -> tuple[str, str]:
        assert obj_type in {"entity", "property"}
        prefix = prefix + "<...>"

        if obj_type == "entity":
            index = self.entity_index
            obj_type_plural = "entities"
        else:
            index = self.property_index
            obj_type_plural = "properties"

        if isinstance(index, PrefixIndex):
            index_info = "keyword prefix index"
            dist_info = "number of keyword matches"
        else:
            assert isinstance(index, QGramIndex)
            if index.distance == "ied":
                dist = "substring"
            else:
                dist = "prefix"
            index_info = "character-level n-gram index"
            dist_info = f"{dist} distance"

        # only lowercase ascii + space, non-empty, up to 128 characters
        regex = r"[a-z0-9 ]{1,128}"
        failure = ""
        if failures:
            failed = "\n".join(failures)
            failure = f"""
The following search queries were already tried but unsuccessful. If there \
is no other sensible search query to try, output one of these again to \
indicate that the search should be stopped at this point:
{failed}
"""
            regex += "|(?:" + "|".join(re.escape(f) for f in failures) + ")"

        prompt = f"""\
Generate a search query for the next {obj_type} to continue the SPARQL \
query with. The search query will be executed over a {index_info} containing \
possible next {obj_type_plural}. It should be short and concise, retrieving \
{obj_type_plural} by {dist_info}. The question to be answered and \
the current SPARQL prefix are given below.

Question:
{question.strip()}

SPARQL prefix over {self.kg}:
{prefix}
{failure}
{obj_type.capitalize()} index search query:
"""
        return prompt, regex

    def get_sparql_prompt(
        self,
        question: str,
        prefix: str,
        examples: list[tuple[str, str]] | None = None,
        failures: set[str] | None = None
    ) -> Chat:
        if prefix == "":
            prefix = "Empty prefix"

        def _ex_prompt(q: str):
            return f"""\
Generate a SPARQL query over {self.kg} to answer the given question.

Question:
{q.strip()}

SPARQL query:
"""

        failure = ""
        if failures:
            failed = "\n".join(failures)
            failure = f"""
The following continuations were already tried but unsuccessful. If there \
is no other sensible continuation to try, output one of these again to \
indicate that the search should be stopped at this point:
{failed}
"""
        prompt = f"""\
Continue the SPARQL prefix over {self.kg} to answer the question \
until the end of the SPARQL query or the next entity or property index \
search via <|kge|> or <|kgp|>.

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
                    "text": _ex_prompt(q)
                },
                {
                    "role": "assistant",
                    "text": s
                }
            ])

        # add actual question
        messages.append({
            "role": "user",
            "text": prompt
        })
        return messages


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
            **{
                short: long
                for long, short in WIKIDATA_PROPERTY_VARIANTS.items()
            }
        })

    def build_alternatives_from_data(
        self,
        data: list[tuple[str, set[str]]]
    ) -> list[Alternative]:
        map = {}
        alternatives: list[Alternative] = []

        for line, variants in data:
            label, _, syns, wid, desc = line.rstrip("\r\n").split("\t")

            if wid not in map:
                map[wid] = variants
                alternative = Alternative(
                    label,
                    wid,
                    None,
                    [s for s in syns.split(";") if s != ""],
                    [desc]
                )
                alternatives.append(alternative)
            else:
                map[wid].update(variants)

        for alternative in alternatives:
            alternative.variants = sorted(map[alternative.identifier])

        return alternatives


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


def search_token_from_obj_type(obj_type: str) -> str:
    if obj_type == "entity":
        return "<|kge|>"
    elif obj_type == "property":
        return "<|kgp|>"
    else:
        raise ValueError(f"unknown object type '{obj_type}'")


def obj_type_from_search(token: str) -> str:
    if token == "<|kge|>":
        return "entity"
    elif token == "<|kgp|>":
        return "property"
    else:
        raise ValueError(f"unknown search token '{token}'")


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


def load_index_and_mapping(
    index_dir: str,
    index_type: str,
    mapping_cls: Type[Mapping] = Mapping,
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
    mapping = mapping_cls.load(
        index,
        os.path.join(index_dir, index_type, "index.mapping")
    )
    return index, mapping
