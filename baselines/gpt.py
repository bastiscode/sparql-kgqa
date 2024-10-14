import argparse
import json
import os
from pprint import pformat
from typing import Type
from pydantic import BaseModel

from openai import OpenAI
from search_index.index import SearchIndex


from sparql_kgqa.sparql.utils import QLEVER_URLS
from sparql_kgqa.sparql.utils2 import (
    KgManager,
    WikidataPropertyMapping,
    get_index_dir,
    get_kg_manager,
    load_index_and_mapping
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question",
        type=str,
        help="Question to translate to SPARQL"
    )
    parser.add_argument(
        "-kg",
        "--knowledge-graph",
        type=str,
        choices=list(QLEVER_URLS),
        default="wikidata",
        help="Knowledge graph used"
    )
    parser.add_argument(
        "-i",
        "--index-type",
        type=str,
        choices=["prefix", "qgram"],
        default="prefix",
        help="Index type to use"
    )
    parser.add_argument(
        "-e",
        "--entities",
        type=str,
        default=None,
        help="Path to entity index"
    )
    parser.add_argument(
        "-p",
        "--properties",
        type=str,
        default=None,
        help="Path to property index"
    )
    parser.add_argument(
        "-a",
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "-k",
        "--search-top-k",
        type=int,
        default=10,
        help="Number of top search results to show"
    )
    parser.add_argument(
        "--save-to",
        type=str,
        default=None,
        help="Save the generation process to the given text file"
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Do not include examples in the generation process"
    )
    return parser.parse_args()


def system_message(manager: KgManager) -> dict:
    kg = manager.kg.capitalize()
    prefixes = "\n".join(
        f"{prefix}: {url}>" for prefix, url in
        (manager.prefixes | manager.custom_prefixes).items()
    )
    return {
        "role": "system",
        "content": f"""\
You are a question answering system over the {kg} knowledge graph. \
Your job is to generate SPARQL queries to answer a given user question over \
{kg}.

You can use the following functions to help you generate the \
SPARQL query:
- stop: Stop the SPARQL generation process
- search_entities: Search for entities in the knowledge graph, \
e.g. search_entities("Angela Merkel")
- search_properties: Search for properties in the knowledge graph, \
e.g. search_properties("instance of")
- execute_sparql: Execute a SPARQL query and return the results as a text \
formatted table, \
e.g. execute_sparql("SELECT ?job WHERE {{ wd:Q937 wdt:P106 ?job }}")

For execute_sparql, you can use the following prefixes without explicitly \
defining them:
{prefixes}

You should follow a step-by-step process to generate the SPARQL query:
1. Generate a high-level plan about what you need to do to answer the question
2. Try to find all entities and properties mentioned in the question
3. Iteratively try to find a single SPARQL query answering the question, \
starting with simple queries first and making them more complex as needed
4. Once you have a final working SPARQL query, execute it and formulate your \
answer, then call stop to end the generation process

Important rules:
- Do not make up information that is not present in the knowledge graph. \
Also do not make up identifiers for entities or properties, only use the \
search functions to find them
- After each function call, interpret and think about its results and \
determine how they can be used to help you generate the final SPARQL query
- You can change your initial plan as you go along, but make sure to explain \
why and how you are changing it
- Your SPARQL queries should always include the entities and properties \
themselves, and not only their labels
- Keep refining your SPARQL query if its results are not what you expect, \
e.g. when obvious entries are missing or too many irrelevant entries are \
included
- Do not manually build SPARQL queries using VALUES clauses and identifiers \
from your search results
- Do not stop early if there are still obvious improvements to be made to \
your SPARQL query

DO NOT USE THE wikibase:label SERVICE IN YOU SPARQL QUERIES, AS IT IS NOT \
SPARQL STANDARD AND NOT SUPPORTED BY THE SPARQL ENGINE USED HERE"""
    }


def prompt(question: str, manager: KgManager) -> dict:
    kg = manager.kg.capitalize()
    return {
        "role": "user",
        "content": f"""\
Write a SPARQL query over {kg} to answer the given question.

Question:
{question}"""
    }


def functions() -> list[dict]:
    return [
        {
            "name": "stop",
            "description": "Stop the SPARQL generation process",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "name": "execute_sparql",
            "description": "Execute a SPARQL query and return the results "
            "as a text formatted table",
            "parameters": {
                "type": "object",
                "properties": {
                    "sparql": {
                        "type": "string",
                        "description": "The SPARQL query to execute"
                    }
                },
                "required": ["sparql"],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "name": "search_entities",
            "description": "Search for entities in the knowledge graph",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "name": "search_properties",
            "description": "Search for properties in the knowledge graph",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True
        }
    ]


def execute_fn(
    manager: KgManager,
    fn_name: str,
    fn_args: dict,
    args: argparse.Namespace
) -> str:
    if fn_name == "execute_sparql":
        return execute_sparql(manager, fn_args["sparql"])

    elif fn_name == "search_entities":
        return search_index(
            manager,
            manager.entity_index,
            fn_args["query"],
            top_k=args.search_top_k
        )

    elif fn_name == "search_properties":
        return search_index(
            manager,
            manager.property_index,
            fn_args["query"],
            top_k=args.search_top_k
        )

    else:
        raise ValueError(f"Unknown function: {fn_name}")


def execute_sparql(manager: KgManager, sparql: str) -> str:
    try:
        sparql = manager.fix_prefixes(sparql)
    except Exception as e:
        return "Failed to fix prefixes in SPARQL query, " \
            f"most likely the query is invalid:\n{e}"

    return manager.get_formatted_result(sparql)


def search_index(
    manager: KgManager,
    index: SearchIndex,
    query: str,
    top_k: int = 5
) -> str:
    matches = index.find_matches(query)
    top_k_string = "\n".join(
        f"{i + 1}. {manager.build_alternative(index.get_row(id)).get_string()}"
        for i, (id, _) in enumerate(matches[:top_k])
    )
    return f"""Got {len(matches):,} matches, the top {top_k} are:
{top_k_string}"""


def format_message(message: dict) -> str:
    role = message["role"].upper()
    return f"""\
{"=" * len(role)}
{role}
{"=" * len(role)}
{message["content"]}"""


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


def easy_example(manager: KgManager) -> list[dict]:
    return [
        prompt("What is the capital of France?", manager),
        {
            "role": "assistant",
            "content": """\
Plan:
- find the entity for France
- find the property for capital
- combine them in a SPARQL query""",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_entities",
                        "arguments": json.dumps({"query": "France"}),
                    },
                    "id": "1"
                }
            ]
        },
        {
            "role": "tool",
            "content": search_index(manager, manager.entity_index, "France"),
            "tool_call_id": "1"
        },
        {
            "role": "assistant",
            "content": """\
I identified the entity for France as wd:Q142. Now I will search for the \
property for capital""",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_properties",
                        "arguments": json.dumps({"query": "capital"}),
                    },
                    "id": "2"
                }
            ]
        },
        {
            "role": "tool",
            "content": search_index(
                manager,
                manager.property_index,
                "capital"
            ),
            "tool_call_id": "2"
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
                        "arguments": json.dumps({
                            "sparql": "SELECT ?capital WHERE { wd:Q142 \
wdt:P36 ?capital }"
                        }),
                    },
                    "id": "3"
                }
            ]
        },
        {
            "role": "tool",
            "content": execute_sparql(
                manager,
                "SELECT ?capital WHERE { wd:Q142 wdt:P36 ?capital }"
            ),
            "tool_call_id": "3"
        },
        {
            "role": "assistant",
            "content": """\
The capital of France is Paris""",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "stop",
                        "arguments": "{}",
                    },
                    "id": "4"
                }
            ]
        },
        {
            "role": "tool",
            "content": "Stopping the SPARQL generation process",
            "tool_call_id": "4"
        }
    ]


def examples(manager: KgManager, ignore: bool = False) -> list[dict]:
    if ignore:
        return []
    else:
        return [
            *easy_example(manager)
        ]


def run(args: argparse.Namespace) -> None:
    args = parse_args()

    kg = args.knowledge_graph
    index_dir = get_index_dir()
    if args.entities is None:
        assert index_dir is not None, \
            "SEARCH_INDEX_DIR environment variable must be set if " \
            "--entities is not provided"
        args.entities = os.path.join(index_dir, f"{kg}-entities")

    if args.properties is None:
        assert index_dir is not None, \
            "SEARCH_INDEX_DIR environment variable must be set if " \
            "--properties is not provided"
        args.properties = os.path.join(index_dir, f"{kg}-properties")

    client = OpenAI(api_key=args.api_key)

    ent_index, ent_mapping = load_index_and_mapping(
        args.entities,
        args.index_type,
    )

    prop_index, prop_mapping = load_index_and_mapping(
        args.properties,
        args.index_type,
        WikidataPropertyMapping if kg == "wikidata" else None,
    )

    manager = get_kg_manager(
        kg,
        ent_index,
        prop_index,
        ent_mapping,
        prop_mapping,
    )

    fns = functions()

    api_messages: list = [
        system_message(manager),
        *examples(manager, args.no_examples),
        prompt(args.question, manager)
    ]
    content_messages: list[str] = []
    for msg in api_messages:
        fmt = format_message(msg)
        print(fmt)
        content_messages.append(fmt)

    sparqls = []

    while True:
        response = client.chat.completions.create(
            messages=api_messages,  # type: ignore
            model=args.model,
            tools=[
                {"type": "function", "function": fn, "strict": True}
                for fn in fns
            ],  # type: ignore
            # response_format=response_format(initial),  # type: ignore
            parallel_tool_calls=False
        )  # type: ignore

        choice = response.choices[0]
        api_messages.append(choice.message)
        if choice.message.content:
            fmt = format_message({
                "role": "assistant",
                "content": choice.message.content or ""
            })
            print(fmt)
            content_messages.append(fmt)

        if choice.finish_reason == "stop":
            break

        elif not choice.message.tool_calls:
            continue

        tool_call = choice.message.tool_calls[0]
        fn_name = tool_call.function.name
        if fn_name == "stop":
            api_messages.append({
                "role": "tool",
                "content": "Stopping the SPARQL generation process",
                "tool_call_id": tool_call.id
            })
            break

        fn_args = json.loads(tool_call.function.arguments)
        if fn_name == "execute_sparql":
            sparqls.append(fn_args["sparql"])

        fmt = format_message({
            "role": "tool call",
            "content": f"Calling function {fn_name} with arguments:\n"
            f"{pformat(fn_args, indent=2)}",
        })
        print(fmt)
        content_messages.append(fmt)

        result = execute_fn(manager, fn_name, fn_args, args)
        tool_msg = {
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id
        }
        fmt = format_message(tool_msg)
        print(fmt)
        content_messages.append(fmt)
        api_messages.append(tool_msg)

    if sparqls:
        sparql = sparqls[-1]
        try:
            sparql = manager.fix_prefixes(sparql)
            sparql = manager.prettify(sparql)
        except Exception:
            pass

        fmt = format_message({
            "role": "sparql",
            "content": sparql
        })
        print(fmt)
        content_messages.append(fmt)

    if args.save_to is not None:
        dir = os.path.dirname(args.save_to)
        if dir:
            os.makedirs(dir, exist_ok=True)

        with open(args.save_to, "w") as f:
            f.write("\n".join(content_messages))


if __name__ == "__main__":
    run(parse_args())
