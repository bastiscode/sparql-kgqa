import json
import sys
import random
from typing import Iterator

import torch

from text_utils.api.cli import TextProcessingCli
from text_utils.api.processor import TextProcessor

from sparql_kgqa import version
from sparql_kgqa.api.generator2 import SPARQLGenerator
from sparql_kgqa.api.server2 import SPARQLGenerationServer
from sparql_kgqa.sparql.utils2 import (
    WikidataPropertyMapping,
    load_index_and_mapping
)


class SPARQLGenerationCli(TextProcessingCli):
    text_processor_cls = SPARQLGenerator
    text_processing_server_cls = SPARQLGenerationServer

    def version(self) -> str:
        return version.__version__

    def setup(self) -> TextProcessor:
        gen = super().setup()
        # perform some additional setup
        assert isinstance(gen, SPARQLGenerator)

        gen.set_inference_options(
            sampling_strategy=self.args.sampling_strategy,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            beam_width=self.args.beam_width,
            max_length=self.args.max_length,
            max_new_tokens=self.args.max_new_tokens,
            disable_sparql_constraint=self.args.no_sparql_constraint,
            disable_subgraph_constraint=self.args.no_subgraph_constraint,
            disable_sparql_judgement=self.args.no_sparql_judgement,
            num_examples=self.args.num_examples,
            select_k=self.args.select_k,
            select_max_candidates=self.args.select_max_candidates,
            select_max_aliases=self.args.select_max_aliases,
            select_add_infos=self.args.select_add_info,
            system_message=self.args.system_message,
            force_exact=self.args.force_exact,
        )

        ent_index, ent_mapping = load_index_and_mapping(
            self.args.entities,
            self.args.index_type,
        )
        is_wikidata = self.args.knowledge_graph == "wikidata"
        prop_index, prop_mapping = load_index_and_mapping(
            self.args.properties,
            self.args.index_type,
            # wikidata properties need special mapping
            # because of wdt, p, ps, pq, ... variants
            WikidataPropertyMapping if is_wikidata else None,
        )

        gen.set_kg_indices(
            self.args.knowledge_graph,
            ent_index,
            prop_index,
            ent_mapping,
            prop_mapping,
        )

        if self.args.example_index is not None:
            gen.set_examples(example_index=self.args.example_index)
        elif self.args.examples is not None:
            gen.set_examples(examples=self.args.examples)

        return gen

    def process_iter(
        self,
        processor: TextProcessor,
        iter: Iterator[str]
    ) -> Iterator[str]:
        assert isinstance(processor, SPARQLGenerator)
        jsonl_in = self.args.input_format == "jsonl"

        for output in processor.generate(
            ((json.loads(item) if jsonl_in else item, None) for item in iter),
            sort=not self.args.unsorted,
            show_progress=self.args.progress,
            pretty=self.args.pretty,
        ):
            if self.args.output_format == "jsonl":
                yield json.dumps(output)
            elif isinstance(output, str):
                yield output
            else:
                yield from output


def main():
    parser = SPARQLGenerationCli.parser(
        "SPARQL Generator",
        "Generate SPARQL from natural language queries."
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["greedy", "top_k", "top_p"],
        type=str,
        default="greedy",
        help="Sampling strategy to use during decoding"
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=1,
        help="Beam width to use for beam search decoding"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Restrict to top k tokens during sampling"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Restrict to top p cumulative probability tokens during sampling"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature to use during sampling"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum supported input/output length in tokens"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--input-format",
        choices=["text", "jsonl"],
        default="text",
        help="Whether to treat input files as jsonl or text"
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "jsonl"],
        default="text",
        help="Whether to format output as jsonl or text"
    )
    parser.add_argument(
        "-kg",
        "--knowledge-graph",
        type=str,
        choices=["wikidata", "freebase", "dbpedia", "dblp"],
        default="wikidata",
        help="Knowledge graph to use for generation",
        required=True
    )
    parser.add_argument(
        "--entities",
        type=str,
        required=True,
        help="Directory of entity index"
    )
    parser.add_argument(
        "--properties",
        type=str,
        required=True,
        help="Directory of property index"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        required=True,
        choices=["prefix", "qgram"],
        help="Index type to use for entity and property indices"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="System message to add to the generation process "
        "(only works with chat models)"
    )
    example_group = parser.add_mutually_exclusive_group()
    example_group.add_argument(
        "--example-index",
        type=str,
        default=None,
        help="Path to example index file"
    )
    example_group.add_argument(
        "--examples",
        type=str,
        default=None,
        help="Path to examples file"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of examples to add to the generation process; top_k "
        "for example index, randomly selected for examples file"
    )
    parser.add_argument(
        "--no-sparql-constraint",
        action="store_true",
        help="Whether to remove SPARQL grammar constraint"
    )
    parser.add_argument(
        "--no-subgraph-constraint",
        action="store_true",
        help="Whether to remove SPARQL subgraph constraint"
    )
    parser.add_argument(
        "--no-sparql-judgement",
        action="store_true",
        help="Whether to skip SPARQL judgement step"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Whether to pretty format SPARQL during postprocessing"
    )
    parser.add_argument(
        "--force-exact",
        action="store_true",
        help="Whether to force using an exact terminal-level SPARQL constraint"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for random number generator"
    )
    parser.add_argument(
        "--select-k",
        type=int,
        default=10,
        help="Number of candidates to select from"
    )
    parser.add_argument(
        "--select-max-candidates",
        type=int,
        default=16384,
        help="Maximum number of candidates for which a sub-index is created"
    )
    parser.add_argument(
        "--select-max-aliases",
        type=int,
        default=5,
        help="Maximum number of aliases for each selection candidate"
    )
    parser.add_argument(
        "--select-add-info",
        action="store_true",
        help="Whether to (forcefully) add additional information to selection "
        "candidates"
    )
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    # set default device to auto if not set
    # (different from underlying library which sets a single gpu as default)
    args.device = args.device or "auto"
    # increase recursion limit
    sys.setrecursionlimit(10000)

    SPARQLGenerationCli(args).run()
