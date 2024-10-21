import argparse
import sys
import os
import json
from enum import Enum
from pydantic import BaseModel

from openai import OpenAI
from tqdm import tqdm

from sparql_kgqa.sparql.metrics import f1_score
from sparql_kgqa.sparql.utils2 import (
    QLEVER_URLS,
    TIMEOUT,
    AskResult,
    KgManager,
    find,
    find_all,
    load_kg_manager,
    parse_to_string,
)


DIR = os.path.dirname(os.path.realpath(__file__))
MAX_RESULTS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input file")
    parser.add_argument("--entities", type=str)
    parser.add_argument("--properties", type=str)
    parser.add_argument(
        "--index-type", type=str, choices=["prefix", "qgram"], default="prefix"
    )
    parser.add_argument(
        "-kg",
        "--knowledge-graph",
        type=str,
        choices=list(QLEVER_URLS),
        default="wikidata",
    )
    parser.add_argument(
        "--api-key",
        type=str,
    )
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument(
        "--examples",
        type=str,
    )
    return parser.parse_args()


def system_prompt(manager: KgManager) -> str:
    kg = manager.kg.capitalize()
    return f"""\
You are a question generation model that generates questions from \
SPARQL queries over {kg}. You are given a SPARQL query and its \
natural language version. You then need to generate a question that \
the SPARQL query answers.

Follow this step-by-step process:
1. Think about the SPARQL query and the question it answers:
- understand the SPARQL query and its natural language version
- look at the entities and properties involved in the query
- in the SPARQL query variable names are normalized as ?var1, ?var2, ... \
and string literals are anonymized as "string1", "string2", ...; think \
about meaningful variable names and string literals that could be used \
as a replacement in the context of the query
2. Clean the SPARQL query:
- replace normalized variable names ?var1, ?var2, ... with meaningful \
variable names like ?country, ?population, ...
- replace anonymized string literals "string1", "string2", ... with \
meaningful string literals like "Germany", "Berlin", ...
- remove unnecessary parts of the query, e.g. unused columns or variables
- remove SERVICE clauses for retrieving labels via wikibase:label and replace \
them with rdfs:label and FILTER clauses wrapped in OPTIONAL or \
leave them out entirely; SERVICE is supported by the SPARQL engine
- the cleaned SPARQL query should still return the same results as the \
original query, at least for the main columns and variables
3. Generate the question that the SPARQL query answers:
- the question should be as concise as possible
- the question should not sound like a description of the SPARQL query itself
- avoid verbatim mentions of entities and properties in \
the question as much as possible, but rather ask about the information that the query \
retrieves; exceptions can be made, e.g. for very specific entities or \
properties or when VALUES clauses are used to filter for specific entities
- the question can be formulated in an asking or requesting manner, e.g. \
"What is the population of Germany?" or "number of people living in Germany"
4. Judge the complexity of the SPARQL query:
- simple: few entities, properties, or variables, and almost no advanced \
SPARQL features
- medium: more entities, properties, or variables, and some advanced SPARQL \
features
- complex: many entities, properties, or variables, and many advanced SPARQL \
features
5. Generate up to 3 paraphrases of the question:
- rephrase the question in different ways while keeping the meaning the same
- the paraphrases should be as concise as possible"""


def prompt(sparql: str, natural_sparql: str, examples: list[tuple[str, str]]) -> str:
    prompt = ""
    if examples:
        prompt += """\
Examples of questions and their corresponding natural language SPARQL \
queries:

"""
        prompt += "\n\n".join(
            f"Question:\n{q}\n\nNatural language SPARQL:\n{s}" for q, s in examples
        )
        prompt += "\n\n"

    prompt += f"""\
For the following SPARQL query and its natural language version, do \
the following:
- Think about the SPARQL query and the question it answers
- Clean the SPARQL query
- Generate the question
- Judge the complexity of the SPARQL query
- Generate up to 3 paraphrases of the question

Keep in mind to generate questions that are as concise as possible \
and contain little to none verbatim mentions of entities and properties \
from the SPARQL query. You are given examples above that you can and \
should use as a reference.

SPARQL query:
{sparql}

Natural language SPARQL query:
{natural_sparql}"""
    return prompt


class Complexity(str, Enum):
    simple = "simple"
    medium = "medium"
    complex = "complex"


class Output(BaseModel):
    thought: str
    question: str
    sparql: str
    complexity: Complexity
    paraphrases: list[str]


def prepare_sparql(manager: KgManager, sparql: str) -> str | None:
    try:
        sparql = manager.fix_prefixes(sparql, remove_known=True)
        sparql, inc = manager.replace_iris(sparql)
        if inc:
            return None
    except Exception:
        return None

    return sparql


def remove_service(manager: KgManager, sparql: str) -> str:
    parse = manager.parser.parse(sparql)

    for service in find_all(parse, "ServiceGraphPattern"):
        var_or_iri = service["children"][2]
        pname = find(var_or_iri, "PNAME_LN")
        if pname is None or pname["value"] != "wikibase:label":
            continue

        service.pop("children")

    return parse_to_string(parse)


def generate_question(
    sparql: str,
    manager: KgManager,
    client: OpenAI,
    examples: list[tuple[str, str]],
    args: argparse.Namespace,
) -> dict:
    output: dict = {"raw_sparql": sparql}

    try:
        sparql = manager.fix_prefixes(sparql, remove_known=True)
    except Exception as e:
        output["error"] = {
            "type": "fix_prefixes",
            "message": str(e),
        }
        return output

    output["fixed_sparql"] = sparql
    sparql_no_service = remove_service(manager, sparql)
    output["fixed_sparql_no_service"] = sparql_no_service

    try:
        original_result = manager.execute_sparql(
            sparql_no_service,
            timeout=TIMEOUT,
            max_results=MAX_RESULTS,
        )
    except Exception as e:
        output["error"] = {
            "type": "execute_original_sparql",
            "message": str(e),
        }
        return output

    natural_sparql, inc = manager.replace_iris(sparql)
    output["natural_sparql"] = natural_sparql

    if inc:
        output["error"] = {
            "type": "replace_iris",
            "message": "some entities or properties mentioned in the "
            "SPARQL could not be replaced with their labels",
        }
        return output

    messages = [
        {"role": "system", "content": system_prompt(manager)},
        {"role": "user", "content": prompt(sparql, natural_sparql, examples)},
    ]

    try:
        response = client.beta.chat.completions.parse(
            messages=messages,  # type: ignore
            response_format=Output,
            model=args.model,
            temperature=1.0,
            top_p=0.9,
        )
        message = response.choices[0].message
        content = message.content
        parsed = message.parsed
    except Exception as e:
        output["error"] = {
            "type": "question_generation",
            "message": str(e),
        }
        return output

    if content is None or parsed is None:
        output["error"] = {
            "type": "response_parsing",
            "message": "no response",
        }
        return output

    output["output"] = json.loads(content)
    # some diagnostics
    print(natural_sparql, file=sys.stderr)
    print(parsed.sparql, file=sys.stderr)
    print(parsed.question, file=sys.stderr)
    print(parsed.paraphrases, file=sys.stderr)
    print(parsed.complexity, file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    try:
        cleaned_result = manager.execute_sparql(
            parsed.sparql, timeout=TIMEOUT, max_results=MAX_RESULTS
        )
    except Exception as e:
        output["error"] = {
            "type": "execute_cleaned_sparql",
            "message": str(e),
        }
        return output

    if not isinstance(original_result, AskResult) and len(original_result[1]) == 0:
        output["error"] = {
            "type": "empty_original_result",
            "message": "original SPARQL query returns an empty result",
        }
        return output

    if not isinstance(cleaned_result, AskResult) and len(cleaned_result[1]) == 0:
        output["error"] = {
            "type": "empty_cleaned_result",
            "message": "cleaned SPARQL query returns an empty result",
        }
        return output

    f1 = f1_score(original_result, cleaned_result, exact=False)
    if f1 < 1.0:
        output["error"] = {
            "type": "results_mismatch",
            "message": "results of original and cleaned sparql do not "
            "match (up to leaving out unnecessary columns and variables)",
            "assignment_f1_score": f1,
        }
        return output

    return output


def run(args: argparse.Namespace) -> None:
    if args.input:
        io = open(args.input, "r")
    else:
        io = sys.stdin

    sparqls = []
    for line in io:
        line = line.rstrip("\r\n")
        sparql = json.loads(line)
        sparqls.append(sparql)

    io.close()
    manager = load_kg_manager(
        args.knowledge_graph, args.entities, args.properties, args.index_type
    )

    examples = []
    if not args.examples:
        args.examples = os.path.join(DIR, "examples_raw.jsonl")

    with open(args.examples, "r") as f:
        for line in f:
            line = line.rstrip("\r\n")
            example = json.loads(line)

            sparql = prepare_sparql(manager, example["sparql"])
            if sparql is None:
                continue

            examples.append((example["question"], sparql))

    client = OpenAI(api_key=args.api_key)

    for sparql in tqdm(sparqls, desc="processing", leave=False):
        output = generate_question(sparql, manager, client, examples, args)
        print(json.dumps(output), flush=True)


if __name__ == "__main__":
    run(parse_args())
