import argparse
import sys
import json
from typing import Type
from pydantic import BaseModel

from openai import OpenAI


from sparql_kgqa.sparql.utils2 import (
    QLEVER_URLS,
    Alternative,
    KgManager,
    AskResult,
    SelectResult,
    extract_field,
    load_kg_manager,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q", "--question", type=str, help="Question to translate to SPARQL"
    )
    parser.add_argument(
        "-kg",
        "--knowledge-graph",
        type=str,
        choices=list(QLEVER_URLS),
        default="wikidata",
        help="Knowledge graph used",
    )
    parser.add_argument(
        "-i",
        "--index-type",
        type=str,
        choices=["prefix", "qgram"],
        default="prefix",
        help="Index type to use",
    )
    parser.add_argument(
        "-e", "--entities", type=str, default=None, help="Path to entity index"
    )
    parser.add_argument(
        "-p", "--properties", type=str, default=None, help="Path to property index"
    )
    parser.add_argument(
        "-a",
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key",
    )
    parser.add_argument(
        "-m", "--model", type=str, default="gpt-4o", help="OpenAI model to use"
    )
    parser.add_argument(
        "-fn",
        "--fn-set",
        type=str,
        default="all",
        choices=["all", "execute", "execute_search"],
        help="Set of functions to use",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature to use during decoding",
    )
    parser.add_argument(
        "-tp",
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p for nucleus sampling",
    )
    parser.add_argument(
        "-sk",
        "--search-top-k",
        type=int,
        default=10,
        help="Number of top search results to show",
    )
    parser.add_argument(
        "-lk",
        "--list-top-k",
        type=int,
        default=30,
        help="Number of top list results to show",
    )
    parser.add_argument(
        "-rk",
        "--result-top-bottom-k",
        type=int,
        default=10,
        help="Number of top and bottom rows to show for a SPARQL query result",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Do not include examples in the generation process",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["text", "jsonl"],
        help="Input format for the question",
    )
    parser.add_argument(
        "--input-field",
        type=str,
        help="Field to extract as question from the input JSON object",
    )
    return parser.parse_args()


def system_message(fns: list[dict], manager: KgManager) -> dict:
    kg = manager.kg.capitalize()
    prefixes = "\n".join(
        f"{prefix}: {url}>"
        for prefix, url in (manager.prefixes | manager.custom_prefixes).items()
    )
    functions = "\n".join(
        f"- {fn['name']}: {fn['description']}"
        + (f" E.g., {fn['example']}" if False and "example" in fn else "")
        for fn in fns
    )
    return {
        "role": "system",
        "content": f"""\
You are a question answering system over the {kg} knowledge graph. \
Your job is to generate SPARQL queries to answer a given user question over \
{kg}.

You can use the following functions to help you generate the \
SPARQL query:
{functions}

For execute_sparql, you can further use the following prefixes without \
explicitly defining them:
{prefixes}

You should follow a step-by-step process to generate the SPARQL query:
1. Generate a high-level plan about what you need to do to answer the question. \
This plan can be subject to change as you gather more information.
2. Find all entities and properties needed to answer the question. \
Always use already identified entities and properties to constrain further \
lookups.
3. Iteratively build up a single SPARQL query answering the question, \
starting with simple queries first and making them more complex as needed.
4. Once you have a final working SPARQL query, execute it and formulate your \
answer. Then call stop to end the generation process.

Important rules:
- Do not make up identifiers for entities or properties, but use the \
provided functions to find them.
- When looking for entities or properties, always use the most constraining \
lookup method first, e.g. start with search_constrained before using \
list_constrained, and only use search_entities or search_properties if \
no constraints are available or the other methods failed.
- After each function call, interpret and think about its results and \
determine how they can be used to help you generate the final SPARQL query.
- Your SPARQL queries should always include the entities and properties \
themselves, and not only their labels.
- Keep refining your SPARQL query if its results are not what you expect, \
e.g. if they are empty when they should not be, when obvious entries are missing \
or too many irrelevant entries are included.
- Do not use results from intermediate SPARQL queries directly in your final \
SPARQL query, e.g. by using them in VALUES clauses.
- Do not stop early if there are still obvious improvements to be made to \
your SPARQL query.
- Do not use the wikibase:label service in your SPARQL queries, as it is not \
SPARQL standard and not supported by the SPARQL engine used here.""",
    }


def prompt(question: str, manager: KgManager) -> dict:
    kg = manager.kg.capitalize()
    return {
        "role": "user",
        "content": f"""\
Write a SPARQL query over {kg} to answer the given question.

Question:
{question}""",
    }


def functions(fn_set: str) -> list[dict]:
    fns = [
        {
            "name": "stop",
            "description": "Stop the SPARQL generation process.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "execute_sparql",
            "description": "Execute a SPARQL query and return the results \
as a text formatted table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sparql": {
                        "type": "string",
                        "description": "The SPARQL query to execute",
                    }
                },
                "required": ["sparql"],
                "additionalProperties": False,
            },
            "strict": True,
            "example": 'execute_sparql(sparql="SELECT ?job WHERE { wd:Q937 \
wdt:P106 ?job }")',
        },
    ]
    if fn_set == "execute":
        return fns

    fns.extend(
        [
            {
                "name": "search_entities",
                "description": "Search for entities in the knowledge graph.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
                "example": 'search_entities(query="Angela Merkel")',
            },
            {
                "name": "search_properties",
                "description": "Search for properties in the knowledge graph.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
                "example": 'search_properties(query="instance of")',
            },
        ]
    )
    if fn_set == "execute_search":
        return fns

    fns.extend(
        [
            {
                "name": "list_constrained",
                "description": "List entities, properties or literals \
given known entities, properties, or literals. The constraints must be properly \
formatted IRIs or literals, e.g. wdt:P31 or \
<http://www.wikidata.org/property/direct/P31>.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": ["string", "null"],
                            "description": "An optional known subject entity",
                        },
                        "property": {
                            "type": ["string", "null"],
                            "description": "An optional known property",
                        },
                        "object": {
                            "type": ["string", "null"],
                            "description": "An optional known object entity or literal",
                        },
                        "list_for": {
                            "type": "string",
                            "enum": ["subject", "property", "object"],
                            "description": "What to list (the respective \
constraining parameter should be null)",
                        },
                    },
                    "required": ["subject", "property", "object", "list_for"],
                    "additionalProperties": False,
                },
                "strict": True,
                "example": 'list_constrained(subject="wd:Q937", property="wdt:P106", \
object=None, list_for="object") to get a list of Albert Einstein\'s jobs',
            },
            {
                "name": "search_constrained",
                "description": "Search for entities, properties or literals \
given known entities, properties, or literals. The constraints must be properly \
formatted IRIs or literals, e.g. wdt:P31 or \
<http://www.wikidata.org/property/direct/P31>.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": ["string", "null"],
                            "description": "An optional known subject entity",
                        },
                        "property": {
                            "type": ["string", "null"],
                            "description": "An optional known property",
                        },
                        "object": {
                            "type": ["string", "null"],
                            "description": "An optional known object entity or literal",
                        },
                        "search_for": {
                            "type": "string",
                            "enum": ["subject", "property", "object"],
                            "description": "What to search for (the respective \
constraining parameter should be null)",
                        },
                        "query": {
                            "type": "string",
                            "description": "A search query used to filter \
the search results",
                        },
                    },
                    "required": [
                        "subject",
                        "property",
                        "object",
                        "search_for",
                        "query",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
                "example": 'search_constrained(subject=None, \
property="wdt:P31", object="wd:Q515", search_for="subject", query="Amsterdam") to \
search for all cities that have Amsterdam in their name',
            },
        ]
    )
    return fns


def execute_fn(
    manager: KgManager, fn_name: str, fn_args: dict, args: argparse.Namespace
) -> str:
    if fn_name == "stop":
        return "Stopping the SPARQL generation process"

    elif fn_name == "execute_sparql":
        return execute_sparql(manager, fn_args["sparql"], args.result_top_bottom_k)

    elif fn_name == "search_entities":
        return search_entities(manager, fn_args["query"], args.search_top_k)

    elif fn_name == "search_properties":
        return search_properties(manager, fn_args["query"], args.search_top_k)

    elif fn_name == "search_constrained":
        return list_or_search_constrained(
            manager, fn_args["search_for"], fn_args["query"], fn_args, args.search_top_k
        )

    elif fn_name == "list_constrained":
        return list_or_search_constrained(
            manager, fn_args["list_for"], None, fn_args, args.list_top_k
        )

    else:
        raise ValueError(f"Unknown function: {fn_name}")


def list_or_search_constrained(
    manager: KgManager,
    target_position: str,
    query: str | None,
    args: dict,
    k: int,
    max_results: int = 1024 * 64,
) -> str:
    target_constr = args[target_position]
    if target_constr is not None:
        return f"Cannot look for {target_position} and constrain it to \
{target_constr} at the same time."

    positions = ["subject", "property", "object"]
    fully_specified = all(args[p] is not None for p in positions)
    none_specified = all(args[p] is None for p in positions)
    if fully_specified or none_specified:
        return "At least one and at most two of subject, property, and \
object should be specified at the same time."

    for p in positions:
        constr = args[p]
        if constr is None:
            continue

        if manager.process_iri_or_literal(constr) is None:
            return f"Constraint {constr} for {p} is not a valid iri or \
literal. Consider fixing its format and try again."

    subject_constr = args["subject"] or "?s"
    property_constr = args["property"] or "?p"
    object_constr = args["object"] or "?o"

    select_var = f"?{target_position[0]}"

    sparql = f"""\
SELECT DISTINCT {select_var} WHERE {{
    {subject_constr}
    {property_constr}
    {object_constr}
}}
LIMIT {max_results + 1}"""
    try:
        result = query_sparql(manager, sparql)
    except Exception as e:
        return f"Failed executing SPARQL query\n{sparql}\n" f"with error:\n{e}"

    assert isinstance(result, tuple)
    _, rows = result
    result_set = set(row[0] for row in rows)
    if len(result_set) > max_results:
        if target_position == "property":
            target_position = "propertie"
        return f"Got more than the maximum number of {target_position}s of \
{max_results:,}. Consider constraining further if possible or try \
to use a standalone SPARQL query via the execute_sparql function."

    (entity_map, property_map, other, literal) = manager.parse_autocompletions(
        result_set
    )

    kg = manager.kg.capitalize()

    formatted = []
    if target_position == "subject":
        alts = manager.get_entity_alternatives(id_map=entity_map, query=query, k=k)
        formatted.append(format_alternatives(f"{kg} entities", alts, k))

    elif target_position == "property":
        alts = manager.get_property_alternatives(id_map=property_map, query=query, k=k)
        formatted.append(format_alternatives(f"{kg} properties", alts, k))
        alts = manager.get_temporary_index_alternatives(data=other, query=query, k=k)
        formatted.append(format_alternatives("other properties", alts, k))

    elif target_position == "object":
        alts = manager.get_entity_alternatives(id_map=entity_map, query=query, k=k)
        formatted.append(format_alternatives(f"{kg} entities", alts, k))
        alts = manager.get_temporary_index_alternatives(data=literal, query=query, k=k)
        formatted.append(format_alternatives("literals", alts, k))

    return "\n\n".join(formatted)


def query_sparql(manager: KgManager, sparql: str) -> AskResult | SelectResult:
    try:
        sparql = manager.fix_prefixes(sparql)
    except Exception as e:
        raise RuntimeError(f"Failed to fix prefixes:\n{e}")

    return manager.execute_sparql(sparql)


def execute_sparql(manager: KgManager, sparql: str, k: int) -> str:
    try:
        result = query_sparql(manager, sparql)
    except Exception as e:
        return f"Failed executing SPARQL query\n{sparql}\n" f"with error:\n{e}"

    return manager.format_sparql_result(result, show_top_bottom_rows=k)


def format_alternatives(name: str, alternatives: list[Alternative], k: int) -> str:
    if len(alternatives) == 0:
        return f"No {name} found"

    top_k_string = "\n".join(
        f"{i + 1}. {alt.get_string()}" for i, alt in enumerate(alternatives[:k])
    )
    return f"Top {k} {name}:\n{top_k_string}"


def search_entities(manager: KgManager, query: str, k: int) -> str:
    alts = manager.get_entity_alternatives(query=query, k=k)
    kg = manager.kg.capitalize()
    return format_alternatives(f"{kg} entities", alts, k)


def search_properties(manager: KgManager, query: str, k: int) -> str:
    alts = manager.get_property_alternatives(query=query, k=k)
    kg = manager.kg.capitalize()
    return format_alternatives(f"{kg} properties", alts, k)


def format_message(message: dict) -> str:
    role = message["role"].upper()
    return f"""\
{"=" * len(role)}
{role}
{"=" * len(role)}
{message["content"]}"""


def format_fn_call(fn_name: str, fn_args: dict) -> str:
    fn_args_string = "\n".join(f"{key} => {value}" for key, value in fn_args.items())
    return f"Calling function {fn_name} with arguments:\n{fn_args_string}"


class Initial(BaseModel):
    plan: str
    action: str


class Continuation(BaseModel):
    thought: str
    plan_change: str | None
    action: str


def response_format(initial: bool) -> Type[BaseModel]:
    if initial:
        return Initial
    else:
        return Continuation


def wikidata_examples(manager: KgManager, args: argparse.Namespace) -> list[dict]:
    if args.fn_set != "all":
        return []

    return [
        prompt("What is the capital of France?", manager),
        {
            "role": "assistant",
            "content": """\
Plan:
- find the entity for France
- find the property for capital given the entity
- combine them in a SPARQL query""",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_entities",
                        "arguments": json.dumps({"query": "France"}),
                    },
                    "id": "1",
                }
            ],
        },
        {
            "role": "tool",
            "content": search_entities(manager, "France", args.search_top_k),
            "tool_call_id": "1",
        },
        {
            "role": "assistant",
            "content": """\
I identified the entity for France as wd:Q142. Let's search for properties \
of France related to capital.""",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_constrained",
                        "arguments": json.dumps(
                            {
                                "subject": "wd:Q142",
                                "property": None,
                                "object": None,
                                "list_for": "property",
                                "query": "capital",
                            }
                        ),
                    },
                    "id": "2",
                }
            ],
        },
        {
            "role": "tool",
            "content": list_or_search_constrained(
                manager,
                "property",
                "capital",
                {
                    "subject": "wd:Q142",
                    "property": None,
                    "object": None,
                },
                args.search_top_k,
            ),
            "tool_call_id": "2",
        },
        {
            "role": "assistant",
            "content": """\
I identified the property for capital as wdt:P36. Now I will combine them \
in a SPARQL query.""",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "execute_sparql",
                        "arguments": json.dumps(
                            {
                                "sparql": "SELECT ?capital WHERE { wd:Q142 \
wdt:P36 ?capital }"
                            }
                        ),
                    },
                    "id": "3",
                }
            ],
        },
        {
            "role": "tool",
            "content": execute_sparql(
                manager,
                "SELECT ?capital WHERE { wd:Q142 wdt:P36 ?capital }",
                args.result_top_bottom_k,
            ),
            "tool_call_id": "3",
        },
        {
            "role": "assistant",
            "content": "The capital of France is Paris",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "stop",
                        "arguments": "{}",
                    },
                    "id": "4",
                }
            ],
        },
        {
            "role": "tool",
            "content": "Stopping the SPARQL generation process",
            "tool_call_id": "4",
        },
    ]


def examples(manager: KgManager, args: argparse.Namespace) -> list[dict]:
    if args.no_examples:
        return []
    elif manager.kg == "wikidata":
        return wikidata_examples(manager, args)
    else:
        return []


def run(args: argparse.Namespace) -> None:
    args = parse_args()

    if args.question is None:
        inputs = [line.rstrip("\r\n") for line in sys.stdin]
    else:
        inputs = [args.question]

    if args.input_format == "jsonl":
        inputs = [json.loads(input) for input in inputs]
        if args.input_field is not None:
            inputs = [extract_field(ipt, args.input_field) for ipt in inputs]

    client = OpenAI(api_key=args.api_key)

    manager = load_kg_manager(
        args.knowledge_graph,
        args.entities,
        args.properties,
        args.index_type,
    )

    fns = functions(args.fn_set)
    system_msg = system_message(fns, manager)
    example_msgs = examples(manager, args)

    for ipt in inputs:
        generate_sparql(client, manager, fns, system_msg, example_msgs, ipt, args)


def generate_sparql(
    client: OpenAI,
    manager: KgManager,
    fns: list[dict],
    system_msg: dict,
    examples: list[dict],
    question: str,
    args: argparse.Namespace,
) -> None:
    api_messages: list[dict] = [
        system_msg,
        *examples,
        prompt(question, manager),
    ]

    for msg in api_messages:
        print(format_message(msg), file=sys.stderr)
        for tool_call in msg.get("tool_calls", []):
            fn_name = tool_call["function"]["name"]
            fn_args = json.loads(tool_call["function"]["arguments"])
            msg = {"role": "tool call", "content": format_fn_call(fn_name, fn_args)}
            print(format_message(msg), file=sys.stderr)

    sparqls = []
    while True:
        try:
            response = client.chat.completions.create(
                messages=api_messages,  # type: ignore
                model=args.model,
                tools=[
                    {"type": "function", "function": fn} for fn in fns
                ],  # type: ignore
                parallel_tool_calls=False,
                temperature=args.temperature,
                top_p=args.top_p,
            )  # type: ignore
        except Exception as e:
            print(f"Failed to generate response: {e}", file=sys.stderr)
            api_messages.append({"error": f"Failed to generate response: {e}"})
            break

        choice = response.choices[0]
        msg = choice.message.to_dict()
        api_messages.append(msg)
        if choice.message.content:
            print(format_message(msg), file=sys.stderr)

        if choice.finish_reason == "stop":
            if sparqls:
                break

            msg = {
                "role": "user",
                "content": "No SPARQL query was generated yet. \
Please continue.",
            }
            api_messages.append(msg)
            print(format_message(msg), file=sys.stderr)
            continue

        elif not choice.message.tool_calls:
            continue

        tool_call = choice.message.tool_calls[0]
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)

        msg = {
            "role": "tool call",
            "content": format_fn_call(fn_name, fn_args),
        }
        print(format_message(msg), file=sys.stderr)

        result = execute_fn(manager, fn_name, fn_args, args)
        tool_msg = {"role": "tool", "content": result, "tool_call_id": tool_call.id}
        print(format_message(tool_msg), file=sys.stderr)
        api_messages.append(tool_msg)

        if fn_name == "execute_sparql":
            sparqls.append(fn_args["sparql"])
        elif fn_name == "stop":
            break

    if sparqls:
        sparql = sparqls[-1]
        try:
            sparql = manager.fix_prefixes(sparql)
            sparql = manager.prettify(sparql)
        except Exception:
            pass
    else:
        sparql = ""

    print(format_message({"role": "sparql", "content": sparql}), file=sys.stderr)
    print(
        json.dumps(
            {
                "sparql": sparql,
                "messages": api_messages,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run(parse_args())
