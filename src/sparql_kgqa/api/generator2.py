from typing import Any, Iterable, Iterator
import logging
import random

import torch
from torch import nn

from text_utils import data, tokenization, grammar
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

from qgram_index import QGramIndex

from sparql_kgqa.model import (
    Model,
    PretrainedDecoder,
    model_from_config,
    peft_model_from_config
)
from sparql_kgqa.sparql.utils import (
    SimilarityIndex,
    load_examples,
    prettify,
)
from sparql_kgqa.sparql.utils2 import (
    KgManager,
    Mapping,
    WikidataManager,
    flatten,
    partition_by,
    run_parallel,
)

LOGGER = logging.getLogger(__name__)
_BASE_URL = ""
_NAME_TO_ZIP = {}

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
        self._top_p = 0.95
        self._use_cache = False
        self._full_outputs = False
        self._max_length = None

        # qgram index options
        self._k = 5  # number of candidates to return
        self._delta: int | None = None  # maximum edit distance
        # maximum size where sub index is built
        self._max_candidates: int | None = 1024

        # SPARQL stuff
        self._exact = self.cfg["inference"].get("exact", False)
        self._force_exact = False
        self._sparql_constraint: None | grammar.LR1Constraint = None
        self._disable_subgraph_constraint = False
        self._disable_sparql_constraint = False
        self._manager: None | KgManager = None
        self._is_chat = self.cfg["inference"].get(
            "chat_template", None
        ) is not None

        self.model = self.model.compile(
            **self.cfg["inference"].get("compile", {})
        )

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
        question: str,
        examples: Examples | None = None,
    ) -> data.InferenceData:
        assert self._manager is not None, "kg indices not set"
        if examples is None and self._example_index is not None:
            examples = self._example_index.top_k(
                question,
                self._num_examples
            )  # type: ignore
        elif examples is None and self._examples is not None:
            examples = random.sample(
                self._examples,
                min(self._num_examples, len(self._examples))
            )
        else:
            examples = []

        assert examples is not None

        prompt = self._manager.get_sparql_prompt(question, examples)
        prompt = self._chat_format(prompt)
        return data.InferenceData(
            prompt,
            {
                "question": question,
                "prompt": prompt,
            }
        )

    def _chat_format(
        self,
        text: str
    ) -> str:
        chat_template = self.cfg["inference"].get("chat_template", None)
        if chat_template is None:
            return text

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
        return s

    @torch.inference_mode()
    def _partial_inference(
        self,
        beams: list[Beam] | list[list[Beam]],
        stop_fn: inference_utils.StopFn
    ) -> Iterator[list[list[Beam]]]:
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

        def update_fn(beam: Beam, token_id: int, log_p: float) -> Beam | None:
            beam = Beam.from_beam(beam, token_id, log_p)

            # advance constraint if given
            if "const" in beam.info:
                const = beam.info["const"]
                if const.is_invalid():
                    # return None if constraint is invalid
                    return None

                # clone constraint and update it
                const = const.clone()
                const.next(token_id)
                beam.info["const"] = const

            return beam

        logit_fns = [
            inference_utils.constraint_logit_fn(
                lambda beam: beam.info.get("const", None),
                self._eos_token_id
            )
        ]

        if self._sampling_strategy == "greedy":
            sample_fn = inference_utils.greedy()
        elif self._sampling_strategy == "top_k":
            assert self._top_k >= self._beam_width, \
                "top k must be greater than or equal to beam width"
            logit_fns.append(inference_utils.top_k_masking(self._top_k))
            sample_fn = inference_utils.sample()
        else:
            logit_fns.append(inference_utils.nucleus_masking(self._top_p))
            sample_fn = inference_utils.sample()

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
            yield_intermediate=True
        )

    @torch.inference_mode()
    def _live_inference(
        self,
        batch: data.InferenceBatch
    ) -> Iterator[list[list[Beam]]]:
        assert self._manager is not None, "kg indices not set"
        assert self._sparql_constraint is not None, "sparql constraint not set"
        # decode fn gets in token ids and additional kwargs,
        # and return logits over next tokens and additional info

        self._sparql_constraint.reset()

        beams = []
        for token_ids, info in zip(batch.token_ids(), batch.infos()):
            beam = Beam(
                token_ids,
                [0.0] * len(token_ids),
                {
                    "question": info["question"],
                    "prompt": info["prompt"],
                    "initial_length": len(token_ids),
                    "last": len(token_ids),
                    "decoded": "",
                    "sparql": "",
                    "guess": "",
                    "seen_sparql": set(),
                }
            )
            if not self._disable_sparql_constraint:
                beam.info["const"] = self._sparql_constraint.clone()

            beams.append([beam])

        def eos_stop_fn(beam: Beam) -> bool:
            return (
                beam.token_ids[-1] == self._eos_token_id
                or len(beam) >= self.max_length
            )

        def pattern_stop_fn(beam: Beam) -> bool:
            decoded = self.tokenizer.de_tokenize(
                beam.token_ids[beam.info["last"]:]
            )
            if eos_stop_fn(beam):
                beam.info["decoded"] += decoded
                beam.info["sparql"] += decoded
                beam.info["guess"] = ""
                return True

            assert self._manager is not None, "kg indices not set"
            match = next(self._manager.pattern.finditer(decoded), None)
            if match is None:
                return False

            part = decoded[:match.start()]
            beam.info["decoded"] += part
            beam.info["sparql"] += part
            beam.info["guess"] = decoded[match.start():]
            return True

        outputs = [[] for _ in range(len(batch))]
        while any(beam for beam in beams):
            for beams in self._partial_inference(beams, pattern_stop_fn):
                yield beams

            # filter out beams that stopped because of eos
            for idx in range(len(batch)):
                stop, keep = partition_by(beams[idx], eos_stop_fn)
                outputs[idx].extend(stop)
                beams[idx] = keep

            alternatives = self._infer_alternatives(beams, eos_stop_fn)
            for idx in range(len(batch)):
                if len(alternatives[idx]) == 0:
                    # keep last non-empty beams
                    outputs[idx].extend(beams[idx])

            beams = alternatives

        yield outputs

    def _infer_alternatives(
        self,
        beams: list[list[Beam]],
        stop_fn: inference_utils.StopFn
    ) -> list[list[Beam]]:
        assert self._manager is not None, "kg indices not set"
        batch_size = len(beams)
        options = (self._k, self._delta, self._max_candidates)
        flat_beams = [
            (idx, beam)
            for idx, beams_ in enumerate(beams)
            for beam in beams_
        ]

        prefixes = [
            beam.info["sparql"] + beam.info["guess"]
            for _, beam in flat_beams
        ]
        results = run_parallel(
            self._manager.get_alternatives,
            ((prefix, *options) for prefix in prefixes)
        )

        keep, _ = partition_by(
            zip(flat_beams, results),
            lambda item: item[1] is not None
        )

        select_beams = []
        for (idx, beam), result in keep:
            assert result is not None
            alts, obj_type, guess = result
            prompt, regex = self._manager.get_alternatives_prompt_and_regex(
                beam.info["question"],
                beam.info["decoded"],
                obj_type,
                guess,
                alts
            )
            prompt = self._chat_format(prompt)
            token_ids = self.tokenizer.tokenize(
                prompt,
                self._is_chat
            ).token_ids
            select_beam = Beam(
                token_ids,
                [0.0] * len(token_ids),
                {
                    "beam": beam,
                    "idx": idx,
                    "prompt": prompt,
                    "alternatives": alts,
                    "obj_type": obj_type,
                    "const": grammar.RegexConstraint(
                        regex,
                        self._continuations
                    ),
                    "initial_length": len(token_ids),
                }
            )
            select_beams.append(select_beam)

        *_, selections = self._partial_inference(
            select_beams,
            stop_fn
        )

        output_beams = [[] for _ in range(batch_size)]
        for selection in flatten(selections):
            selected = self.tokenizer.de_tokenize(
                selection.token_ids[selection.info["initial_length"]:]
            )
            LOGGER.debug(selection.info["prompt"] + selected)
            try:
                selected = self._manager.parse_result(
                    selection.info["alternatives"],
                    selection.info["obj_type"],
                    selected
                )
            except Exception as e:
                LOGGER.debug(f"error parsing result '{selected}': {e}")
                selected = None

            if selected is None:
                continue

            identifier, name = selected
            beam: Beam = selection.info["beam"]
            sparql = beam.info["sparql"] + identifier
            if sparql in beam.info["seen_sparql"]:
                continue
            beam.info["seen_sparql"].add(sparql)
            # build new input from prompt and parts
            decoded = beam.info["decoded"] + name

            token_ids = self.tokenizer.tokenize(
                beam.info["prompt"] + decoded,
                self._is_chat
            ).token_ids
            updated_beam = Beam(
                token_ids,
                [0.0] * len(token_ids),
                {k: v for k, v in beam.info.items()},
            )

            updated_beam.info["decoded"] = decoded
            updated_beam.info["sparql"] = sparql
            updated_beam.info["last"] = len(token_ids)
            if not self._disable_sparql_constraint:
                updated_beam.info["const"] = beam.info["const"].clone()
                updated_beam.info["const"].reset(decoded.encode())

            idx = selection.info["idx"]
            output_beams[idx].append(updated_beam)

        return output_beams

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
        entity_index: QGramIndex | tuple[str, str],
        property_index: QGramIndex | tuple[str, str],
        entity_mapping: Mapping | str | None = None,
        property_mapping: Mapping | str | None = None
    ) -> None:
        if kg == "wikidata":
            self._manager = WikidataManager(
                entity_index,
                property_index,
                entity_mapping,
                property_mapping  # type: ignore
            )
            self._sparql_constraint = self._manager.get_constraint(
                self._continuations,
                self._exact or self._force_exact
            )
        else:
            raise ValueError(f"kg {kg} not supported")

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
        select_k: int = 5,
        select_delta: int | None = None,
        select_max_candidates: int | None = 1024,
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
        self._k = select_k
        self._delta = select_delta
        self._max_candidates = select_max_candidates

    def generate(
        self,
        inputs: Iterable[tuple[str, Examples | None]],
        batch_size: int = 16,
        batch_max_tokens: int | None = None,
        sort: bool = True,
        num_threads: int | None = None,
        show_progress: bool = False,
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
            assert self._manager is not None, "kg indices not set"
            processed = []
            for output in outputs[0]:
                input = output.info["prompt"]
                generated = output.info["sparql"]

                if pretty:
                    generated = self._manager.prettify(
                        generated,
                        is_prefix=True
                    )

                if self._full_outputs:
                    generated = input + generated

                if not return_candidates:
                    # return top candidate only
                    return generated

                processed.append(generated)

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
        postprocess: bool = True,
        pretty: bool = False
    ) -> Iterator[list[str]]:
        input = self._prepare_input(query, examples)
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

            sparqls = []
            for beam in beams:
                init = 0 if self._full_outputs else beam.info["initial_length"]
                sparql_prefix = self.tokenizer.de_tokenize(
                    beam.token_ids[init:]
                )
                if pretty:
                    try:
                        sparql_prefix = prettify(
                            sparql_prefix,
                            self._sparql_parser,
                            is_prefix=True
                        )
                    except Exception:
                        pass

                sparqls.append(sparql_prefix)

            yield sparqls

        if not postprocess:
            return

        assert last is not None, "should not happen"

        sparqls = []
        for beam in last:
            init = beam.info["initial_length"]
            sparql = self.tokenizer.de_tokenize(beam.token_ids[init:])
            if self._full_outputs:
                input = self.tokenizer.de_tokenize(beam.token_ids[:init])
            else:
                input = ""

            try:
                sparql = input + postprocess_sparql_query(
                    sparql,
                    self._sparql_parser,
                    beam.info["entities"],
                    beam.info["properties"],
                    self._prefixes,
                    pretty
                )
            except Exception:
                sparql = input + sparql

            sparqls.append(sparql)

        yield sparqls
