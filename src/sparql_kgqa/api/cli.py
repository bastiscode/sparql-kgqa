import json
import random
import argparse
from typing import Iterable, Iterator

from tqdm import tqdm

from text_utils.api.cli import TextProcessingCli
from text_utils.api.processor import TextProcessor

from sparql_kgqa import version
from sparql_kgqa.api.generator import SPARQLGenerator
from sparql_kgqa.api.server import SPARQLGenerationServer
from sparql_kgqa.sparql.utils import load_examples


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
            force_exact=self.args.force_exact
        )

        for kg in self.args.knowledge_graph or []:
            gen.set_indices(*kg)

        return gen

    def format_output(self, output: str) -> Iterable[str]:
        if self.args.output_format == "jsonl":
            return [json.dumps(output)]
        return [output]

    def process_iter(
        self,
        processor: TextProcessor,
        iter: Iterator[str]
    ) -> Iterator[str]:
        assert isinstance(processor, SPARQLGenerator)
        if self.args.examples is not None:
            examples = load_examples(self.args.examples)
        else:
            examples = []

        for item in tqdm(
            iter,
            desc="Generating SPARQL",
            disable=not self.args.progress or self.args.process
        ):
            if self.args.input_format == "jsonl":
                item = json.loads(item)

            sampled = random.sample(
                examples,
                min(len(examples), self.args.num_examples)
            )

            *_, final = processor.generate(
                item,
                info=self.args.info,
                examples=sampled,
                preprocessed=self.args.preprocessed,
                postprocess=not self.args.no_postprocessing,
                pretty=self.args.pretty
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
    parser.add_argument(
        "--no-postprocessing",
        action="store_true",
        help="Whether to skip postprocessing"
    )

    class KgAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) != 5:  # type: ignore
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
        nargs=5,
        action=KgAction,
        metavar=(
            "ENTITY_INDEX",
            "RELATION_INDEX",
            "ENTITY_PREFIXES",
            "RELATION_PREFIXES",
            "KG_NAME"
        ),
        help="Add knowledge graph to the generation process"
    )
    parser.add_argument(
        "--examples",
        type=str,
        default=None,
        help="Path to examples file"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of examples to add to the generation process, randomly "
        "selected if there are more examples than this number"
    )
    parser.add_argument(
        "--no-sparql-constraint",
        action="store_true",
        help="Whether to remove SPARQL grammar constraint"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Whether to pretty format SPARQL during postprocessing"
    )
    parser.add_argument(
        "--force-exact",
        action="store_true",
        help="Whether to force using a exact SPARQL constraint"
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
    # set default device to auto if not set
    # (different from underlying library which sets a single gpu as default)
    args.device = args.device or "auto"
    SPARQLGenerationCli(args).run()
