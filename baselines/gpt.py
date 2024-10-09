import argparse
import json
from pprint import pformat

from openai import OpenAI
from search_index.index import SearchIndex


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
        "entities",
        type=str,
        help="Path to entity index"
    )
    parser.add_argument(
        "properties",
        type=str,
        help="Path to property index"
    )
    parser.add_argument(
        "index_type",
        type=str,
        choices=["prefix", "qgram"],
        help="Index type"
    )
    parser.add_argument(
        "kg",
        type=str,
        choices=list(QLEVER_URLS),
        help="Knowledge graph used"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
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
{kg}. You can use the following functions to help you generate the \
SPARQL query:
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
3. Iteratively try to combine all previous steps into a single SPARQL query, \
starting with simple queries first and making them more complex as needed
4. Once you have a final working SPARQL query, execute it and return the \
results

Important rules:
- Do not make up information that is not present in the knowledge graph. \
Also do not make up identifiers for entities or properties, only use the \
search functions to find them
- After each function call, interpret and think about its results and \
determine how they can be used to help you generate the final SPARQL query
- You can change your initial plan as you go along, but make sure to explain \
why you are changing it and how the new plan is better than the old one
- Your SPARQL queries should always include the entities and properties \
themselves, and not only their labels
- Do not use the SERVICE keyword in SPARQL queries, as it is not supported \
by the used SPARQL endpoint
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
            }
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
            }
        }
    ]


def execute_fn(manager: KgManager, fn_name: str, args: dict) -> str:
    if fn_name == "execute_sparql":
        return execute_sparql(manager, args["sparql"])

    elif fn_name == "search_entities":
        return search_index(manager, manager.entity_index, args["query"])

    elif fn_name == "search_properties":
        return search_index(manager, manager.property_index, args["query"])

    else:
        raise ValueError(f"Unknown function: {fn_name}")


def execute_sparql(manager: KgManager, sparql: str) -> str:
    try:
        sparql = manager.fix_prefixes(sparql)
    except Exception:
        return "Failed to fix prefixes in SPARQL query"

    return manager.get_formatted_result(sparql)


def search_index(
    manager: KgManager,
    index: SearchIndex,
    query: str,
    top_k: int = MAX_RESULTS
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


def run(args: argparse.Namespace) -> None:
    args = parse_args()
    client = OpenAI(api_key=args.api_key)

    ent_index, ent_mapping = load_index_and_mapping(
        args.entities,
        args.index_type,
    )

    prop_index, prop_mapping = load_index_and_mapping(
        args.properties,
        args.index_type,
        WikidataPropertyMapping if args.kg == "wikidata" else None,
    )

    manager = get_kg_manager(
        args.kg,
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
