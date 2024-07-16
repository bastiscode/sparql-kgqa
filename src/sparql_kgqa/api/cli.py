import json
import sys
import random
import argparse
from typing import Iterator

import torch

from text_utils.api.cli import TextProcessingCli
from text_utils.api.processor import TextProcessor

from sparql_kgqa import version
from sparql_kgqa.api.generator import SPARQLGenerator
from sparql_kgqa.api.server import SPARQLGenerationServer


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
            use_cache=self.args.kv_cache,
            full_outputs=self.args.full_outputs,
            disable_sparql_constraint=self.args.no_sparql_constraint,
            disable_subgraph_constraint=self.args.no_subgraph_constraint,
            num_examples=self.args.num_examples,
            system_message=self.args.system_message,
            force_exact=self.args.force_exact,
        )

        for (
            ent_data,
            ent_index,
            prop_data,
            prop_idx,
            kg
        ) in self.args.knowledge_graph or []:
            gen.set_kg_indices(
                kg,
                (ent_data, ent_index),
                (prop_data, prop_idx),
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
            ((
                json.loads(item) if jsonl_in else item,
                None,
                self.args.preprocessed,
            ) for item in iter),
            batch_size=self.args.batch_size,
            batch_max_tokens=self.args.batch_max_tokens,
            sort=not self.args.unsorted,
            show_progress=self.args.progress,
            postprocess=not self.args.no_postprocessing,
            pretty=self.args.pretty,
            return_candidates=self.args.return_candidates,
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
        "--kv-cache",
        action="store_true",
        help="Whether to use key and value caches during decoding"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum supported input/output length in tokens"
    )
    parser.add_argument(
        "-full",
        "--full-outputs",
        action="store_true",
        help="Whether to return input and generated text as output "
        "(default is only generated text)"
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
        "--return-candidates",
        action="store_true",
        help="Whether to return full candidate outputs "
        "(only best generated output by default)"
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="Whether input is already preprocessed"
    )
    parser.add_argument(
        "--no-postprocessing",
        action="store_true",
        help="Whether to skip postprocessing"
    )

    class KgAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) != 5:  # type: ignore
                parser.error(f"{option_string} requires exactly five strings")
            # Retrieve the current list or create a new one if it's None
            current_list = getattr(namespace, self.dest, None) or []
            # Append the tuple of three strings
            current_list.append(tuple(values))  # type: ignore
            # Save the list back to the namespace
            setattr(namespace, self.dest, current_list)

    parser.add_argument(
        "-kg",
        "--knowledge-graph",
        type=str,
        nargs=5,
        action=KgAction,
        metavar=(
            "ENT_DATA_DIR",
            "ENT_INDEX_DIR",
            "PROP_DATA_DIR",
            "PROP_INDEX_DIR",
            "KG_NAME"
        ),
        help="Add knowledge graph to the generation process"
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
