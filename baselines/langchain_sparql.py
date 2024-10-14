import argparse

from langchain.chains import GraphSparqlQAChain
from langchain_community.graphs import RdfGraph
from langchain_openai import ChatOpenAI

from sparql_kgqa.sparql.utils import QLEVER_URLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question",
        type=str,
        help="Question to ask",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default=QLEVER_URLS["wikidata"],
        help="URL of the SPARQL endpoint",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    graph = RdfGraph(
        query_endpoint=args.url,
        store_kwargs={"returnFormat": "tsv"},
    )
    chain = GraphSparqlQAChain.from_llm(
        ChatOpenAI(model=args.model),
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True
    )
    chain.run(args.question)


if __name__ == "__main__":
    run(parse_args())
