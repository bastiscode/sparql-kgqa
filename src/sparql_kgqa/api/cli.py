import json
import argparse
from typing import Iterable, Iterator

from tqdm import tqdm

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
            full_outputs=self.args.full_outputs
        )

        for kg in self.args.knowledge_graph:
            gen.set_indices(*kg)

        return gen

    def format_output(self, output: str) -> Iterable[str]:
        if self.args.file is not None and self.args.output_format == "jsonl":
            return [json.dumps(output)]

        return [output]

    def process_iter(
        self,
        processor: SPARQLGenerator,
        iter: Iterator[str]
    ) -> Iterator[str]:
        for item in tqdm(
            iter, 
            desc="Generating SPARQL", 
            disable=not self.args.progress or self.args.process
        ):
            if self.args.file is not None and self.args.input_format == "jsonl":
                item = json.loads(item)

            *_, final = processor.generate_live(
                item,
                info=self.args.info,
                preprocessed=self.args.preprocessed
            )

            yield final


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
        "--info",
        type=str,
        default=None,
        help="Additional information for SPARQL generation"
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="Whether input is already preprocessed"
    )

    class KgAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) != 3:  # type: ignore
                parser.error(f"{option_string} requires exactly three strings")
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
        nargs=3,
        action=KgAction,
        metavar=("ENTITY_INDEX", "RELATION_INDEX", "KG_NAME"),
        help="Add knowledge graph to the generation process"
    )
    parser.add_argument(
        "--no-sparql-constraint",
        action="store_true",
        help="Whether to remove SPARQL grammar constraint"
    )
    args = parser.parse_args()
    # set default device to auto if not set
    # (different from underlying library which sets a single gpu as default)
    args.device = args.device or "auto"
    SPARQLGenerationCli(args).run()
