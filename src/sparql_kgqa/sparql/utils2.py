from collections import Counter
import random
import logging
import re
import pickle
import uuid
from importlib import resources
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Iterator, Type, TypeVar

from text_utils import grammar
from qgram_index import QGramIndex

from sparql_kgqa.sparql.utils import (
    _find_all,
    _find,
    _parse_to_string,
    ask_to_select,
    prettify,
    query_qlever,
    SelectResult
)

LOGGER = logging.getLogger(__name__)
CLEAN_PATTERN = re.compile(r"\s+", flags=re.MULTILINE)


def clean(s: str) -> str:
    return CLEAN_PATTERN.sub(" ", s).strip()


def _load_sparql_grammar(
    entity_variants: set[str] | None = None,
    property_variants: set[str] | None = None
) -> tuple[str, str]:
    sparql_grammar = resources.read_text(
        "sparql_kgqa.sparql.grammar",
        "sparql.y"
    )
    sparql_lexer = resources.read_text(
        "sparql_kgqa.sparql.grammar",
        "sparql2.l"
    )
    ent = '|'.join(re.escape(v) for v in entity_variants or set())
    sparql_lexer.replace(
        "ENTITY_VARIANTS ''",
        f"ENTITY_VARIANTS '{ent}'"
    )
    prop = '|'.join(re.escape(v) for v in property_variants or set())
    sparql_lexer.replace(
        "PROPERTY_VARIANTS ''",
        f"PROPERTY_VARIANTS '{prop}'"
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
        if self.aliases:
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
    def __init__(self, map: dict[str, int]) -> None:
        self.map = map

    @classmethod
    def from_qgram_index(cls, index: QGramIndex) -> "Mapping":
        map = {}
        for i in range(len(index)):
            data = index.get_data_by_idx(i)
            obj_id = data.split("\t")[3]
            assert obj_id not in map, f"obj_id {obj_id} is not unique"
            map[obj_id] = i
        return cls(map)

    def save(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            pickle.dump(self.map, f)

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "rb") as f:
            return cls(pickle.load(f))

    def __getitem__(self, key: str) -> int:
        return self.map[key]

    def normalize(self, iri: str) -> tuple[str, str | None] | None:
        return iri, None

    def denormalize(self, key: str, variant: str | None) -> str | None:
        return key

    def default_variants(self) -> set[str]:
        return set()

    def __contains__(self, key: str) -> bool:
        return key in self.map


class WikidataPropertyMapping(Mapping):
    NORM_PREFIX = "<http://www.wikidata.org/entity/"

    def __init__(self, map: dict[str, int]) -> None:
        super().__init__(map)
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
        entity_index: QGramIndex | tuple[str, str],
        property_index: QGramIndex | tuple[str, str],
        entity_mapping: Mapping | str | None = None,
        property_mapping: Mapping | str | None = None,
        parser: grammar.LR1Parser | None = None,
    ):
        self.kg = kg

        if isinstance(entity_index, tuple):
            self.entity_index = QGramIndex.load(*entity_index)
        else:
            self.entity_index = entity_index

        if entity_mapping is None:
            self.entity_mapping = self.entity_mapping_cls.from_qgram_index(
                self.entity_index
            )
        elif isinstance(entity_mapping, str):
            self.entity_mapping = self.entity_mapping_cls.load(entity_mapping)
        else:
            assert isinstance(entity_mapping, self.entity_mapping_cls), \
                f"entity_mapping is not of type {self.entity_mapping_cls}"
            self.entity_mapping = entity_mapping

        if isinstance(property_index, tuple):
            self.property_index = QGramIndex.load(*property_index)
        else:
            self.property_index = property_index

        if property_mapping is None:
            self.property_mapping = self.property_mapping_cls.from_qgram_index(
                self.property_index
            )
        elif isinstance(property_mapping, str):
            self.property_mapping = self.property_mapping_cls.load(
                property_mapping
            )
        else:
            assert isinstance(property_mapping, self.property_mapping_cls), \
                f"property_mapping is not of type {self.property_mapping_cls}"
            self.property_mapping = property_mapping

        if parser is None:
            self.parser = self.get_parser()
        else:
            self.parser = parser

        variants = "|".join(
            re.escape(v)
            for v in
            self.entity_mapping.default_variants()
            | self.property_mapping.default_variants()
        )
        self.pattern = re.compile(
            rf"<\|kg([ep])\|>(.*?)(?: \(({variants})\))?<\|kg\1\|>"
        )

    def get_parser(self) -> grammar.LR1Parser:
        sparql_grammar, sparql_lexer = _load_sparql_grammar(
            self.entity_mapping.default_variants(),
            self.property_mapping.default_variants()
        )
        return grammar.LR1Parser(
            sparql_grammar,
            sparql_lexer,
        )

    def get_constraint(
        self,
        continuations: list[bytes],
        exact: bool
    ) -> grammar.LR1Constraint:
        sparql_grammar, sparql_lexer = _load_sparql_grammar(
            self.entity_mapping.default_variants(),
            self.property_mapping.default_variants()
        )
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
        limit: int | None = None
    ) -> tuple[set[str] | None, str, tuple[str, str | None]] | None:
        """
        Autocomplete the SPARQL query prefix,
        run it against Qlever and return the entities or
        properties that can come next.
        Assumes that the prefix is a valid SPARQL query prefix,
        otherwise throws an exception.
        """
        matches = list(self.pattern.finditer(prefix))
        match = matches[-1]
        prefix = prefix[:match.start()]
        obj_type = "entity" if match.group(1) == "e" else "property"
        name = match.group(2)
        variant = match.group(3)
        if variant == "":
            variant = None
        guess = (name, variant)

        parse, _ = self.parser.prefix_parse(
            prefix.encode(),
            skip_empty=False,
            collapse_single=True
        )

        if _find(
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
            # which means that there are no constraints
            return None, obj_type, guess

        # determine current position in the query:
        # subject, predicate or object
        triple_blocks = list(_find_all(
            parse,
            "TriplesSameSubjectPath",
        ))
        if len(triple_blocks) == 0:
            # without triples the knowledge graph can not be
            # constrained
            return None, obj_type, guess

        last_triple = triple_blocks[-1]
        # the last triple block
        assert len(last_triple["children"]) == 2
        first, second = last_triple["children"]
        assert second["name"] == "PropertyListPathNotEmpty"

        var = uuid.uuid4().hex
        assert len(second["children"]) == 3
        if _parse_to_string(second["children"][1]) != "":
            # subject can always be any iri
            return None, obj_type, (name, variant)

        elif _parse_to_string(second["children"][0]) != "":
            # object
            second["children"][1] = {"name": "VAR1", "value": f"?{var}"}

        elif _parse_to_string(first) != "":
            # property
            second["children"][0] = {"name": "VAR1", "value": f"?{var}"}
            obj_var = uuid.uuid4().hex
            second["children"][1] = {"name": "VAR1", "value": f"?{obj_var}"}

        else:
            assert "unexpected case"

        # fix all future brackets
        for item in _find_all(
            parse,
            {"{", "}", "(", ")", "."},
        ):
            item["value"] = item["name"]

        if limit is not None:
            # find solution modifier and add limit clause
            sol_mod = _find(parse, "SolutionModifier")
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

        prefix = _parse_to_string(parse)

        select = ask_to_select(prefix, self.parser,
                               var=f"?{var}", distinct=True)
        if select is not None:
            prefix = select
        else:
            # query is not an ask query, replace
            # the selected vars with our own
            parse = self.parser.parse(
                prefix, skip_empty=False, collapse_single=False)
            sel_clause = _find(parse, "SelectClause", skip={"SubSelect"})
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

        result = query_qlever(
            prefix,
            self.parser,
            self.kg,
            qlever_endpoint,
            timeout=10.0
        )
        assert isinstance(result, SelectResult)
        uris = set()
        for res in result.results:
            record = res[var]
            if record is None or record.data_type != "uri":
                continue
            uris.add("<" + record.value + ">")
        return uris, obj_type, guess

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

        prologue = _find(parse, "Prologue")
        assert prologue is not None

        base_decls = list(_find_all(prologue, "BaseDecl"))

        exist = {}
        for prefix_decl in _find_all(prologue, "PrefixDecl"):
            assert len(prefix_decl["children"]) == 3
            short = prefix_decl["children"][1]["value"].split(":", 1)[0]
            long = prefix_decl["children"][2]["value"][:-1]
            exist[short] = long

        seen = set()
        for iri in _find_all(parse, "IRIREF"):
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

        for pfx in _find_all(
            parse,
            {"PNAME_NS", "PNAME_LN"},
            skip={"Prologue"}
        ):
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

        return _parse_to_string(parse) + rest_str

    def replace_iris(
        self,
        sparql: str,
        replacement: str = "label",
    ) -> tuple[str, bool]:
        assert replacement in [
            "label",
            "synonyms"
        ]

        parse = self.parser.parse(
            sparql,
            skip_empty=True,
            collapse_single=False
        )
        incomplete = False

        for obj in _find_all(parse, "iri", skip={"Prologue"}):
            child = obj["children"][0]
            is_in_kg = False
            if child["name"] == "PrefixedName":
                val = child["children"][0]["value"]
                # convert to long form
                pfx, val = val.split(":", 1)
                if pfx not in self.custom_prefixes:
                    continue
                val = self.custom_prefixes[pfx] + val + ">"
                is_in_kg = True
            elif child["name"] == "IRIREF":
                val = child["value"]
                is_in_kg = next(filter(
                    lambda pfx: val.startswith(pfx),
                    self.custom_prefixes.values()
                ), None) is not None
            else:
                continue

            if not is_in_kg:
                continue

            norm = self.entity_mapping.normalize(val)
            map = self.entity_mapping
            index = self.entity_index
            obj_type = "KGE"
            if norm is None or norm[0] not in map:
                norm = self.property_mapping.normalize(val)
                map = self.property_mapping
                index = self.property_index
                obj_type = "KGP"

            if norm is None or norm[0] not in map:
                incomplete = True
                continue

            key, variant = norm
            data = index.get_data_by_idx(map[key])
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

            child["name"] = obj_type
            if obj_type == "KGE":
                label = f"<|kge|>{label}<|kge|>"
            else:
                label = f"<|kgp|>{label}<|kgp|>"

            child.pop("children", None)
            child["value"] = label

        return _parse_to_string(parse), incomplete

    def replace_entities_and_properties(self, sparql: str) -> str:
        parse = self.parser.parse(
            sparql,
            skip_empty=True,
            collapse_single=True
        )

        for obj in _find_all(
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
            name = "KGE"
            if norm is None or norm[0] not in map:
                # fallback to property
                norm = self.property_mapping.normalize(val)
                map = self.property_mapping
                index = self.property_index
                name = "KGP"

            # if norm is still none or key not in map, continue
            if norm is None or norm[0] not in map:
                continue

            key, variant = norm
            data = index.get_data_by_idx(map[key])
            label = data.split("\t")[0]
            if variant is not None:
                label += f" ({variant})"

            if name == "KGE":
                obj["value"] = f"<|kge|>{label}<|kge|>"
            else:
                obj["value"] = f"<|kgp|>{label}<|kgp|>"
            obj["name"] = name

        return _parse_to_string(parse)

    def alternatives_from_data(
        self,
        data: list[tuple[str, set[str]]],
    ) -> list[Alternative]:
        raise NotImplementedError

    def get_alternatives(
        self,
        prefix: str,
        k: int,
        delta: int | None = None,
        max_candidates: int | None = None,
        endpoint: str | None = None
    ) -> tuple[list[Alternative], str, tuple[str, str | None]] | None:
        try:
            result = self.autocomplete_prefix(
                prefix,
                endpoint,
                max_candidates + 1
                if max_candidates is not None else None,
            )
        except Exception as e:
            LOGGER.debug(
                f"autocomplete_prefix failed for prefix '{prefix}': "
                f"{e}"
            )
            result = None

        if result is None:
            return None

        select_result, obj_type, guess = result
        if obj_type == "entity":
            index = self.entity_index
            map = self.entity_mapping
        else:
            index = self.property_index
            map = self.property_mapping

        data: list[tuple[str, set[str]]] = []
        if (
            select_result is None
            or len(select_result) > (max_candidates or len(select_result))
        ):
            # select result being None means that there is no way
            # to constrain / filter the knowledge graph with the
            # current prefix, just search in the full index
            # with the guess;
            # we also do this if the number of results is greater
            # than max_results, because creating an extra qgram index
            # for that would be too expensive
            for i, _ in index.find_matches(guess[0], delta)[:k]:
                data.append((
                    index.get_data_by_id(i),
                    map.default_variants()
                ))

        elif k < len(select_result):
            # build a sub index and find matches in it
            indices = []
            valid_variants = []
            for iri in select_result:
                norm = map.normalize(iri)
                if norm is None:
                    continue
                iri, variant = norm
                if iri not in map:
                    continue
                valid_variants.append(variant)
                indices.append(map[iri])

            sub_index = index.sub_index_by_indices(indices)
            for i, _ in sub_index.find_matches(guess[0], delta)[:k]:
                variant = valid_variants[sub_index.get_idx_by_id(i)]
                data.append((
                    sub_index.get_data_by_id(i),
                    set() if variant is None else {variant}
                ))

        else:
            # we have less than k result, just get all of them
            for iri in select_result:
                norm = map.normalize(iri)
                if norm is None:
                    continue
                iri, variant = norm
                if iri not in map:
                    continue
                data.append((
                    index.get_data_by_idx(map[iri]),
                    set() if variant is None else {variant}
                ))

        alternatives = self.alternatives_from_data(data)
        return alternatives, obj_type, guess

    def parse_result(
        self,
        alternatives: list[Alternative],
        obj_type: str,
        result: str
    ) -> tuple[str, str] | None:
        num, name = result.split(".", 1)
        idx = int(num) - 1
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

        if obj_type == "entity":
            map = self.entity_mapping
            name = f"<|kge|>{name.strip()}<|kge|>"
        else:
            map = self.property_mapping
            name = f"<|kgp|>{name.strip()}<|kgp|>"

        denorm = map.denormalize(alternative.identifier, variant)
        if denorm is None:
            return None
        elif alternative.identifier not in map:
            return None
        else:
            return denorm, name.strip()

    def get_sparql_prompt(
        self,
        question: str,
        examples: list[tuple[str, str]] | None = None
    ) -> str:
        def _prompt(q: str):
            return f"""\
Question:
{q.strip()}

SPARQL query over {self.kg}:
"""

        inputs = []
        for q, s in examples or []:
            try:
                s = self.fix_prefixes(self.replace_entities_and_properties(s))
            except Exception:
                # skip invalid examples
                continue
            inputs.append(_prompt(q) + s)

        # add actual question
        inputs.append(_prompt(question))
        return "\n\n".join(inputs)

    def get_alternatives_prompt_and_regex(
        self,
        question: str,
        prefix: str,
        obj_type: str,
        guess: tuple[str, str | None],
        alternatives: list[Alternative],
        add_none_alternative: bool = True,
        max_aliases: int = 5,
        add_infos: bool = False
    ) -> tuple[str, str]:
        assert obj_type in {"entity", "property"}
        first = obj_type[0]
        prefix = prefix + f"<kg{first}>...<kg{first}>"

        counts = Counter(alternative.label for alternative in alternatives)
        alt_strings = []
        alt_regexes = []
        for i, alternative in enumerate(alternatives):
            i_str = f"{i + 1}. "
            alt_strings.append(i_str + alternative.get_string(
                max_aliases,
                # add info to non unique labels
                add_infos or counts[alternative.label] > 1
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

        name, variant = guess
        if variant is not None:
            name = f"{name} ({variant})"

        prompt = f"""\
Question:
{question.strip()}

SPARQL query prefix over {self.kg}:
{prefix}

Current {obj_type} guess:
{name}

Actual {obj_type} alternatives:
{alt_string}

The most fitting {obj_type} alternative for continuing the SPARQL \
query to answer the given question is:
"""
        return prompt, "(?:" + "|".join(alt_regexes) + ")"


class WikidataManager(KgManager):
    property_mapping_cls = WikidataPropertyMapping

    def __init__(
        self,
        entity_index: QGramIndex | tuple[str, str],
        property_index: QGramIndex | tuple[str, str],
        entity_mapping: Mapping | str | None = None,
        property_mapping: WikidataPropertyMapping | str | None = None,
        parser: grammar.LR1Parser | None = None,
    ):
        super().__init__(
            "wikidata",
            entity_index,
            property_index,
            entity_mapping,
            property_mapping,
            parser,
        )
        # add wikidata specific prefixes
        self.custom_prefixes.update({
            "wd": "<http://www.wikidata.org/entity/",
            **{
                short: long
                for long, short in WIKIDATA_PROPERTY_VARIANTS.items()
            }
        })

    def alternatives_from_data(
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
                    syns.split(";"),
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


T = TypeVar("T")


def flatten(
    iter: Iterable[Iterable[T]]
) -> Iterator[T]:
    for sub_iter in iter:
        yield from sub_iter


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
