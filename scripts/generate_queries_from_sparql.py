import argparse
import sys
import os
import json
from pprint import pprint, pformat
from enum import Enum
from pydantic import BaseModel

from openai import OpenAI
from tqdm import tqdm

from sparql_kgqa.sparql.metrics import assignment_f1_score, calculate_f1_score
from sparql_kgqa.sparql.utils2 import (
    QLEVER_URLS,
    KgManager,
    find,
    find_all,
    load_kg_manager,
    parse_to_string
)


DIR = os.path.dirname(os.path.realpath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input file"
    )
    parser.add_argument(
        "--entities",
        type=str
    )
    parser.add_argument(
        "--properties",
        type=str
    )
    parser.add_argument(
        "--index-type",
        type=str,
        choices=["prefix", "qgram"],
        default="prefix"
    )
    parser.add_argument(
        "-kg",
        "--knowledge-graph",
        type=str,
        choices=list(QLEVER_URLS),
        default="wikidata"
    )
    parser.add_argument(
        "--api-key",
        type=str,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o"
    )
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
1. Clean the SPARQL query:
- remove unnecessary parts of the query, e.g. unused columns or variables
- remove SERVICE clauses for retrieving labels via wikibase:label and replace \
them with FILTER clauses; they are not supported by the SPARQL engine
- the cleaned SPARQL query should still return the same results as the \
original query, at least for the main columns and variables
2. Think about the question that the SPARQL query answers:
- understand the SPARQL query and its natural language version
- look at the entities and properties involved in the query
3. Generate the question that the SPARQL query answers:
- the question should be as concise as possible
- the question should not sound like a description of the SPARQL query itself
- DO NOT include entities and properties from the query verbatim in the \
question, but rather ask about the information that the query retrieves
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


def prompt(
    sparql: str,
    natural_sparql: str,
    examples: list[tuple[str, str]]
) -> str:
    prompt = ""
    if examples:
        prompt += """\
Examples of questions and their corresponding natural language SPARQL \
queries:

"""
        prompt += "\n\n".join(
            f"Question:\n{q}\n\nNatural language SPARQL:\n{s}"
            for q, s in examples
        )
        prompt += "\n\n"

    prompt += f"""\
For the following SPARQL query and its natural language version, do \
the following:
- Clean the SPARQL query
- Think about the question that the SPARQL query answers
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
        sparql = manager.fix_prefixes(
            sparql,
            remove_known=True
        )
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
        args.knowledge_graph,
        args.entities,
        args.properties,
        args.index_type
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

    outputs = []
    for sparql in tqdm(sparqls, desc="processing", leave=False):
        outputs.append({
            "raw_sparql": sparql,
        })
        try:
            sparql = manager.fix_prefixes(
                sparql,
                remove_known=True
            )
        except Exception as e:
            outputs[-1]["error"] = {
                "type": "fix_prefixes",
                "message": str(e),
            }
            continue

        outputs[-1]["fixed_sparql"] = sparql

        natural_sparql, inc = manager.replace_iris(sparql)
        outputs[-1]["natural_sparql"] = natural_sparql

        if inc:
            outputs[-1]["error"] = {
                "type": "replace_iris",
                "message": "some entities or properties mentioned in the "
                "SPARQL could not be replaced with their labels",
            }
            continue

        messages = [
            {"role": "system", "content": system_prompt(manager)},
            {
                "role": "user",
                "content": prompt(sparql, natural_sparql, examples)
            }
        ]

        try:
            response = client.beta.chat.completions.parse(
                messages=messages,  # type: ignore
                response_format=Output,
                model=args.model
            )
            message = response.choices[0].message
            content = message.content
            parsed = message.parsed
        except Exception as e:
            print(
                "Failed to generate completion for messages\n"
                f"{pformat(messages)}\nwith error:\n{e}"
            )
            outputs[-1]["error"] = {
                "type": "question_generation",
                "message": str(e),
            }
            continue

        if content is None or parsed is None:
            print(f"Failed to parse completion for messages\n{messages}")
            outputs[-1]["error"] = {
                "type": "response_parsing",
                "message": "no response",
            }
            continue

        sparql_no_service = remove_service(manager, sparql)
        outputs[-1]["fixed_sparql_no_service"] = sparql_no_service
        outputs[-1]["model"] = json.loads(content)

        f1, new_err, org_err = calculate_f1_score(
            sparql_no_service,
            parsed.sparql,
            manager,
            allow_empty_target=False,
            exact=False
        )
        if f1 is None:
            outputs[-1]["error"] = {
                "type": "result_comparison_error",
                "original_sparql_message": org_err,
                "cleaned_sparql_message": new_err,
            }
            continue

        elif f1 < 1.0:
            outputs[-1]["error"] = {
                "type": "result_comparison_unequal",
                "message": "results of original and cleaned sparql do not "
                "match (up to leaving out unnecessary columns and variables)",
                "assignment_f1_score": f1,
            }
            continue

    pprint(outputs)


if __name__ == "__main__":
    run(parse_args())