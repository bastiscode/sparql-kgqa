from typing import Any, Iterable, Iterator
import re
import random
import os
import copy

import torch
from torch import nn

from text_utils import data, tokenization, continuations
from text_utils.api.processor import ModelInfo, TextProcessor
from text_utils.api.utils import (
    Device,
    device_info,
    get_devices,
)
from text_utils.inference import (
    utils as inference_utils,
    beam_search
)
from text_utils.inference.utils import Beam
from text_utils.constraints import ContinuationConstraint

from sparql_kgqa.model import (
    Model,
    PretrainedDecoder,
    model_from_config,
    peft_model_from_config
)
from sparql_kgqa.sparql.utils import (
    SimilarityIndex,
    load_examples,
    replace_entities_and_properties,
    subgraph_constraint,
    general_prefixes,
    load_prefixes,
    load_sparql_constraint,
    load_sparql_parser,
    postprocess_sparql_query,
    preprocess_natural_language_query
)

_BASE_URL = ""
_NAME_TO_ZIP = {}

Const = str | tuple[str, str, bool]
ContIndex = continuations.MmapContinuationIndex
Examples = list[tuple[str, str]]


class SPARQLGenerator(TextProcessor):
    task = "SPARQL Generation"

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        return []

    @classmethod
    def _model_url(cls, model: str) -> str:
        return f"{_BASE_URL}/{_NAME_TO_ZIP[model]}"

    @property
    def name(self) -> str:
        return self.cfg["experiment"]["name"]

    @classmethod
    def _model_from_config(
        cls,
        cfg: dict[str, Any],
        device: Device
    ) -> nn.Module:
        model = model_from_config(cfg["model"])
        assert isinstance(model, PretrainedDecoder)
        peft = cfg["train"].get("peft", None)
        if peft is not None:
            model = peft_model_from_config(model, peft)
        return model

    @property
    def max_length(self) -> int:
        cfg_max_length = self.cfg["inference"].get("max_length", 512)
        return min(
            self._max_length or cfg_max_length,
            cfg_max_length
        )

    def __init__(
        self,
        model: Model,
        cfg: dict[str, Any],
        device: Device
    ) -> None:
        super().__init__(model, cfg, device)
        assert isinstance(model, PretrainedDecoder)
        self.logger.debug(f"got model config:\n{self.cfg['model']}")
        self.logger.info(
            f"running {self.name} text generator "
            f"on devices {[device_info(d) for d in self.devices]}"
        )
        self.tokenizer = tokenization.Tokenizer.from_config(
            self.cfg["inference"]["tokenizer"]
        )

        # some options for inference
        self._eos_token = self.cfg["inference"]["eos"]
        self._eos_token_id = self.tokenizer.token_to_id(
            self._eos_token
        )
        assert self._eos_token_id is not None, \
            f"token {self._eos_token} not in tokenizer"

        # continuations are the postprocessed tokens from the vocab
        # (already sorted by token id)
        self._continuations = self.tokenizer.get_continuations(initial=False)
        self._sampling_strategy = "greedy"
        self._beam_width = 1
        self._temp = 1.0
        self._top_k = 5
        self._use_cache = False
        self._full_outputs = False
        self._max_length = None

        # SPARQL stuff
        self._exact = self.cfg["inference"].get("exact", False)
        self._force_exact = False
        self._sparql_constraint = load_sparql_constraint(
            [],
            self._continuations,
            self._exact or self._force_exact
        )
        self._disable_sparql_constraint = False
        self._disable_subgraph_constraint = False
        self._sparql_parser = load_sparql_parser([])
        self._is_chat = self.cfg["inference"].get(
            "chat_template", None
        ) is not None
        self._prefixes = general_prefixes()

        self.model = self.model.compile(
            **self.cfg["inference"].get("compile", {})
        )

        self._entity_indices = {}
        self._property_indices = {}
        self._example_index: SimilarityIndex | None = None
        self._examples: Examples | None = None
        self._num_examples: int = 3
        self._default_system_message: str | None = self.cfg["inference"].get(
            "system_message", None
        )
        self._system_message: str | None = None

    def to(self, device: Device) -> "SPARQLGenerator":
        self.devices = get_devices(device)
        if self.cfg["model"].get("device_map", None) is not None:
            return self
        assert isinstance(self.model, Model)
        self.model = self.model.distribute(self.devices)
        return self

    def _prepare_input(
        self,
        text: str,
        examples: Examples | None = None,
        preprocessed: bool = False
    ) -> data.InferenceData:
        if not preprocessed:
            if examples is None and self._example_index is not None:
                examples = self._example_index.top_k(
                    text,
                    self._num_examples
                )  # type: ignore
            elif examples is None and self._examples is not None:
                examples = random.sample(
                    self._examples,
                    min(self._num_examples, len(self._examples))
                )

            text = preprocess_natural_language_query(
                text,
                list(self._entity_indices),
                examples
            )

        chat_template = self.cfg["inference"].get("chat_template", None)
        if chat_template is not None:
            assert "user" in chat_template["roles"], \
                "chat template must have a user role"

            s = chat_template.get("start", "")

            system_message = (
                self._system_message or self._default_system_message
            )
            if system_message is not None:
                assert "system" in chat_template["roles"], \
                    "chat template must have a system role"
                s += chat_template["roles"]["system"].replace(
                    "{text}",
                    system_message
                )
            s += chat_template["roles"]["user"].replace("{text}", text)
            s += chat_template.get("end", "")
            text = s

        return data.InferenceData(text, {})

    @torch.inference_mode()
    def _live_inference(
        self,
        batch: data.InferenceBatch
    ) -> Iterator[list[list[Beam]]]:
        kgs = list(self._entity_indices)
        kgs = "|".join(re.escape(kg) for kg in kgs)
        START_PATTERN = re.compile(f"<kg(e|p) kg='({kgs})'>")
        END_PATTERN = re.compile("</kg(?:e|p)>")

        # decode fn gets in token ids and additional kwargs,
        # and return logits over next tokens and additional info
        def decode_fn(
            token_ids: torch.Tensor,
            **kwargs: Any
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            assert isinstance(self.model, PretrainedDecoder)
            dec, cache = self.model.decode(
                token_ids,
                kwargs["lengths"],
                kwargs.get("kv_cache", None),
                self._use_cache
            )
            return dec, {"kv_cache": cache}

        def kwargs_update_fn(
            kwargs: dict[str, Any],
            info: dict[str, Any],
            mask: torch.Tensor
        ) -> None:
            kv_cache = info.get("kv_cache", None)
            if kv_cache is None:
                return

            kwargs["kv_cache"] = tuple(
                tuple(c[mask.to(c.device)] for c in cache)
                for cache in info["kv_cache"]
            )

        self._sparql_constraint.reset()

        beams = []
        for token_ids in batch.token_ids():
            beam = Beam(
                token_ids,
                [0.0] * len(token_ids),
                {
                    "index": None,
                    "initial_length": len(token_ids),
                    "start_at": len(token_ids),
                    "sparql_const": None
                    if self._disable_sparql_constraint
                    else self._sparql_constraint.clone(),
                    "index_const": None,
                    "entities": {},
                    "properties": {},
                    "values": {}
                }
            )
            beams.append(beam)

        def stop_fn(beam: Beam) -> bool:
            return beam.token_ids[-1] == self._eos_token_id

        def update_fn(beam: Beam, token_id: int, log_p: float) -> Beam | None:
            beam = Beam.from_beam(beam, token_id, log_p)
            if stop_fn(beam):
                # do not update anything if beam will be stopped anyway
                return beam

            # copy entities and properties between beams
            beam.info["entites"] = copy.deepcopy(beam.info["entities"])
            beam.info["properties"] = copy.deepcopy(beam.info["properties"])

            for const_name in ["index_const", "sparql_const"]:
                const = beam.info[const_name]
                if const is None:
                    continue
                elif const.is_invalid():
                    # return None if constraint is invalid
                    return None

                # clone constraint and update it
                const = const.clone()
                const.next(token_id)
                beam.info[const_name] = const

            entities = beam.info["entities"]
            properties = beam.info["properties"]
            index = beam.info["index"]

            last_decoded = self.tokenizer.de_tokenize(
                beam.token_ids[beam.info["start_at"]:]
            )
            if index is not None:
                match = END_PATTERN.search(last_decoded)
                if match is None:
                    return beam

                name = last_decoded[:match.start()]

                const = beam.info["index_const"]
                assert isinstance(const, ContinuationConstraint)
                obj_id = const.get_value()
                assert obj_id is not None
                obj_type, kg, initial_prefix = index
                name = initial_prefix + name
                if obj_type == "e":
                    if kg not in entities:
                        entities[kg] = {}
                    entities[kg][name] = obj_id
                else:
                    if kg not in properties:
                        properties[kg] = {}
                    properties[kg][name] = obj_id

                beam.info["index"] = None
                beam.info["index_const"] = None
                beam.info["start_at"] = len(beam.token_ids)
                return beam

            match = START_PATTERN.search(last_decoded)
            if match is None or len(self._entity_indices) == 0:
                # no start pattern found or no indices set
                return beam

            obj_type = match.group(1)
            kg = match.group(2)
            initial_prefix = last_decoded[match.end():]
            beam.info["index"] = (obj_type, kg, initial_prefix)
            beam.info["start_at"] = len(beam.token_ids)

            if obj_type == "e":
                kg_index = self._entity_indices[kg]
            else:
                kg_index = self._property_indices[kg]

            if not self._disable_subgraph_constraint:
                full_decoded = self.tokenizer.de_tokenize(
                    beam.token_ids[beam.info["initial_length"]:]
                )
                * _, last = START_PATTERN.finditer(full_decoded)

                try:
                    full_parsed = self._sparql_parser.prefix_parse(
                        full_decoded[:last.start()].encode(),
                        skip_empty=True,
                        collapse_single=True
                    )
                    full_replaced = replace_entities_and_properties(
                        full_parsed,
                        self._sparql_parser,
                        entities,
                        properties
                    )
                except Exception:
                    full_replaced = None

                values = beam.info["values"].get(full_replaced, None)
                if values is None:
                    try:
                        values = subgraph_constraint(
                            full_decoded[:last.end()],
                            self._sparql_parser,
                            entities,
                            properties,
                            self._prefixes,
                            limit=8193
                        )
                    except Exception:
                        # keep none values, which means
                        # no subgraph constraint
                        pass

                if values is not None:
                    beam.info["values"][full_replaced] = values
                    if 0 < len(values) <= 8192:
                        kg_index = kg_index.sub_index_by_values(values)

            try:
                beam.info["index_const"] = ContinuationConstraint(
                    kg_index,
                    initial_prefix.encode(),
                )
            except Exception:
                # initial prefix no valid prefix for kg index
                # should only happen with non exact
                # SPARQL constraint
                return None

            return beam

        logit_fns = [
            inference_utils.constraint_logit_fn(
                lambda beam: (
                    beam.info["index_const"] or beam.info["sparql_const"]
                    if isinstance(beam, Beam) else None
                ),
                self._eos_token_id
            )
        ]

        if self._sampling_strategy == "greedy":
            sample_fn = inference_utils.beam_greedy()
        elif self._sampling_strategy == "top_k":
            assert self._top_k >= self._beam_width, \
                "top k must be greater than or equal to beam width"
            logit_fns.append(inference_utils.top_k_masking(self._top_k))
            sample_fn = inference_utils.beam_sample()
        else:
            logit_fns.append(inference_utils.nucleus_masking(self._top_p))
            sample_fn = inference_utils.beam_sample()

        if self._sampling_strategy != "greedy" and self._temp != 1.0:
            logit_fns.append(inference_utils.temperature_scaling(
                self._temp
            ))

        yield from beam_search(
            decode_fn=decode_fn,
            initial=beams,
            pad_token_id=self.tokenizer.pad_token_id(),
            max_length=self.max_length,
            stop_fn=stop_fn,  # type: ignore
            device=self.devices[0],
            normalize_by_length=True,
            alpha=1.0,
            beam_width=self._beam_width,
            sample_fn=sample_fn,
            candidate_fn=update_fn,
            logit_fns=logit_fns,
            kwargs_update_fn=kwargs_update_fn,
            return_full=True,
            return_incomplete=True,
            yield_intermediate=True
        )

    def set_examples(
        self,
        examples: Examples | str | None = None,
        example_index: SimilarityIndex | str | None = None
    ) -> None:
        if example_index is not None:
            if isinstance(example_index, str):
                example_index = SimilarityIndex.load(example_index)
            self._example_index = example_index
            self._examples = None

        elif examples is not None:
            if isinstance(examples, str):
                examples = load_examples(examples)
            self._examples = examples
            self._example_index = None

        else:
            raise ValueError("either examples or example_index must be set")

    def set_kg_indices(
        self,
        kg: str,
        entities: tuple[str, str] | tuple[ContIndex, dict[str, str]],
        properties: tuple[str, str] | tuple[ContIndex, dict[str, str]],
    ) -> None:
        first, second = entities
        if isinstance(first, str):
            assert isinstance(second, str)
            entity_index = ContIndex.load_with_continuations(
                os.path.join(first, "index.tsv"),
                second,
                self._continuations,
                common_suffix=self.cfg["inference"].get(
                    "entity_suffix",
                    "</kge>"
                )
            )
            entity_prefixes = load_prefixes(
                os.path.join(first, "prefixes.tsv")
            )
        else:
            assert isinstance(second, dict)
            entity_index = first.clone_with_continuations(
                self._continuations
            )
            entity_prefixes = second

        first, second = properties
        if isinstance(first, str):
            assert isinstance(second, str)
            property_index = ContIndex.load_with_continuations(
                os.path.join(first, "index.tsv"),
                second,
                self._continuations,
                common_suffix=self.cfg["inference"].get(
                    "property_suffix",
                    "</kgp>"
                )
            )
            property_prefixes = load_prefixes(
                os.path.join(first, "prefixes.tsv")
            )
        else:
            assert isinstance(second, dict)
            property_index = first.clone_with_continuations(
                self._continuations
            )
            property_prefixes = second

        self._entity_indices[kg] = entity_index
        self._property_indices[kg] = property_index
        self._prefixes.update(entity_prefixes)
        self._prefixes.update(property_prefixes)

        # reload constraint with new kgs
        kgs = list(self._entity_indices)
        self._sparql_constraint = load_sparql_constraint(
            kgs,
            self._continuations,
            self._exact or self._force_exact
        )
        self._sparql_parser = load_sparql_parser(kgs)

    def set_inference_options(
        self,
        sampling_strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.95,
        beam_width: int = 1,
        max_length: int | None = None,
        use_cache: bool = False,
        full_outputs: bool = False,
        disable_sparql_constraint: bool = False,
        disable_subgraph_constraint: bool = False,
        num_examples: int = 3,
        system_message: str | None = None,
        force_exact: bool = False
    ) -> None:
        assert sampling_strategy in ["greedy", "top_k", "top_p"]
        self._sampling_strategy = sampling_strategy
        self._beam_width = beam_width
        self._temp = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._max_length = max_length
        self._use_cache = use_cache
        self._full_outputs = full_outputs
        self._disable_sparql_constraint = disable_sparql_constraint
        self._disable_subgraph_constraint = disable_subgraph_constraint
        self._force_exact = force_exact
        self._num_examples = num_examples
        self._system_message = system_message

    def generate(
        self,
        inputs: Iterable[tuple[str, Examples | None, bool]],
        batch_size: int = 16,
        batch_max_tokens: int | None = None,
        sort: bool = True,
        num_threads: int | None = None,
        show_progress: bool = False,
        postprocess: bool = True,
        pretty: bool = False,
        return_candidates: bool = False
    ) -> Iterator[str | list[str]]:
        def inference_fn(
            batch: data.InferenceBatch
        ) -> list[list[Beam]]:
            *_, last = self._live_inference(batch)
            return last

        def postprocessing_fn(
            items: list[data.InferenceItem],
            outputs: list[list[Beam]],
        ) -> str | list[str]:
            assert len(items) == 1 and len(outputs) == 1
            processed = []
            for output in outputs[0]:
                init = output.info["initial_length"]
                sparql = self.tokenizer.de_tokenize(output.token_ids[init:])
                if self._full_outputs:
                    input = self.tokenizer.de_tokenize(output.token_ids[:init])
                else:
                    input = ""

                if postprocess:
                    try:
                        sparql = postprocess_sparql_query(
                            sparql,
                            self._sparql_parser,
                            output.info["entities"],
                            output.info["properties"],
                            self._prefixes,
                            pretty
                        )
                    except Exception:
                        pass

                if not return_candidates:
                    # return top candidate only
                    return input + sparql

                processed.append(input + sparql)

            return processed

        yield from self._process(
            (self._prepare_input(*input) for input in inputs),
            inference_fn,
            postprocessing_fn,
            "Generating SPARQL queries",
            batch_size,
            batch_max_tokens,
            sort,
            num_threads,
            show_progress=show_progress,
            ignore_special_tokens=self._is_chat
        )

    def generate_live(
        self,
        query: str,
        examples: Examples | None = None,
        preprocessed: bool = False,
        postprocess: bool = True,
        pretty: bool = False
    ) -> Iterator[list[str]]:
        input = self._prepare_input(query, examples, preprocessed)
        batch = next(data.InferenceLoader.from_iterator(
            iter([input]),
            self.cfg["inference"]["tokenizer"],
            self.cfg["inference"].get("window", {"type": "full"}),
            ignore_special_tokens=self._is_chat
        ))

        # tokenize and de_tokenize here to get rid of
        # special tokens and start/end patterns
        token_ids = self.tokenizer.tokenize(input.text).token_ids
        yield [self.tokenizer.de_tokenize(token_ids)]

        last: list[Beam] | None = None
        for output in self._live_inference(batch):
            beams = output[0]
            last = beams

            decoded = []
            for beam in beams:
                init = 0 if self._full_outputs else beam.info["initial_length"]
                decoded.append(self.tokenizer.de_tokenize(
                    beam.token_ids[init:]
                ))

            yield decoded

        if not postprocess:
            return

        assert last is not None, "should not happen"

        decoded = []
        for beam in last:
            init = beam.info["initial_length"]
            sparql = self.tokenizer.de_tokenize(beam.token_ids[init:])
            if self._full_outputs:
                input = self.tokenizer.de_tokenize(beam.token_ids[:init])
            else:
                input = ""

            try:
                output = input + postprocess_sparql_query(
                    sparql,
                    self._sparql_parser,
                    beam.info["entities"],
                    beam.info["properties"],
                    self._prefixes,
                    pretty
                )
            except Exception:
                output = input + sparql

            decoded.append(output)

        yield decoded
