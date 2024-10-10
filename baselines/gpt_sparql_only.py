import argparse
import json
import os
from pprint import pformat

from openai import OpenAI


from sparql_kgqa.sparql.utils import QLEVER_URLS
from sparql_kgqa.sparql.utils2 import (
    KgManager,
    WikidataPropertyMapping,
    get_kg_manager,
    load_index_and_mapping
)


MAX_RESULTS = 5


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
Your job is to generate a SPARQL query to answer a given user question over \
{kg}. You can use the function "execute_sparql" to execute a SPARQL query and \
get its results as a text formatted table, \
e.g. execute_sparql("SELECT ?job WHERE {{ wd:Q937 wdt:P106 ?job }}").

For "execute_sparql", you can use the following prefixes without explicitly \
defining them:
{prefixes}

You should follow a step-by-step process to generate the SPARQL query:
1. Generate a high-level plan about what you need to do to answer the question
2. Iteratively build up your final SPARQL query, starting with simple queries \
first and making them more complex as needed
3. Once you have a final working SPARQL query, execute it and return the \
results

Important rules:
- Do not make up information that is not present in the knowledge graph. \
Also do not make up identifiers for entities or properties, but extract \
them from the knowledge graph using the provided functions
- After each function call, interpret and think about its results and \
determine how they can be used to help you generate the final SPARQL query
- You can change your initial plan as you go along, but make sure to explain \
why you are changing it and how the new plan is better than the old one
- Your SPARQL queries should always include the entities and properties \
themselves, and not only their labels
- Do not use the SERVICE keyword in SPARQL queries, as it is not supported \
by the used SPARQL engine
"""
    }


def prompt(question: str, manager: KgManager) -> dict:
    kg = manager.kg.capitalize()
    return {
        "role": "user",
        "content": f"""\
Answer the following question by generating a SPARQL query over {kg}:
{question}"""
    }


def functions() -> list[dict]:
    return [
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
            }
        },
    ]


def execute_fn(manager: KgManager, fn_name: str, args: dict) -> str:
    if fn_name == "execute_sparql":
        return execute_sparql(manager, args["sparql"])

    else:
        raise ValueError(f"Unknown function: {fn_name}")


def execute_sparql(manager: KgManager, sparql: str) -> str:
    try:
        sparql = manager.fix_prefixes(sparql)
    except Exception:
        return "Failed to fix prefixes in SPARQL query"

    return manager.get_formatted_result(sparql)


def format_message(message: dict) -> str:
    role = message["role"].upper()
    return f"""\
{"=" * len(role)}
{role}
{"=" * len(role)}
{message["content"]}"""


def run(args: argparse.Namespace) -> None:
    args = parse_args()
    client = OpenAI(api_key=args.api_key)

    kg = args.knowledge_graph
    index_dir = os.getenv("SEARCH_INDEX_DIR", None)
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

    ent_index, ent_mapping = load_index_and_mapping(
        args.entities,
        "prefix"
    )

    prop_index, prop_mapping = load_index_and_mapping(
        args.properties,
        "prefix",
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

    system_msg = system_message(manager)
    print(format_message(system_msg))
    prompt_msg = prompt(args.question, manager)
    print(format_message(prompt_msg))

    api_messages: list = [
        system_msg,
        prompt_msg
    ]

    sparqls = []

    while True:
        response = client.chat.completions.create(
            messages=api_messages,  # type: ignore
            model=args.model,
            tools=[
                {"type": "function", "function": fn, "strict": True}
                for fn in fns
            ],  # type: ignore
            parallel_tool_calls=False
        )  # type: ignore

        choice = response.choices[0]
        api_messages.append(choice.message)

        if not choice.message.tool_calls:
            print(format_message({
                "role": "assistant",
                "content": choice.message.content or ""
            }))

            if choice.finish_reason == "stop":
                break

            continue

        for tool_call in choice.message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            if fn_name == "execute_sparql":
                sparqls.append(fn_args["sparql"])

            print(format_message({
                "role": "tool call",
                "content": f"Calling function {fn_name} with arguments:\n"
                f"{pformat(fn_args, indent=2)}",
            }))

            result = execute_fn(manager, fn_name, fn_args)
            tool_msg = {
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id
            }
            print(format_message(tool_msg))
            api_messages.append(tool_msg)

    if sparqls:
        sparql = sparqls[-1]
        try:
            sparql = manager.fix_prefixes(sparql)
            sparql = manager.prettify(sparql)
        except Exception:
            pass

        print(format_message({
            "role": "sparql",
            "content": sparql
        }))


if __name__ == "__main__":
    run(parse_args())
