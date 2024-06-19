from typing import Any, Iterator
import re
import os

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
    search,
    beam_search
)
from text_utils.inference.utils import (
    Beam,
    BeamStopFn,
    DecodeFn,
    MaskUpdateFn,
    StopFn
)
from text_utils.constraints import Constraint, ContinuationConstraint

from sparql_kgqa.model import (
    Model,
    PretrainedDecoder,
    model_from_config,
    peft_model_from_config
)
from sparql_kgqa.sparql.utils import (
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
        cfg_max_length = self.cfg["train"]["data"].get("max_length", 512)
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
        self._eos_token_id = self.tokenizer.special_token_to_id(
            self._eos_token
        )

        # continuations are the tokens from the vocab
        # (already sorted by token id)
        self._continuations = self.tokenizer.get_vocab()
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
            self.tokenizer.get_vocab(),
            self._exact or self._force_exact
        )
        self._disable_sparql_constraint = False
        self._sparql_parser = load_sparql_parser([])

        self.model = self.model.compile(
            **self.cfg["inference"].get("compile", {})
        )

        self._entity_indices = {}
        self._property_indices = {}
        self._prefixes = general_prefixes()

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
        info: str | None = None,
        examples: list[tuple[str, str]] | None = None,
        preprocessed: bool = False
    ) -> data.InferenceData:
        if not preprocessed:
            text = preprocess_natural_language_query(
                text,
                list(self._entity_indices),
                info,
                examples
            )
        else:
            assert info is None, \
                "info must be None if text is already preprocessed"

        if "chat_template" in self.cfg["inference"]:
            template = self.cfg["inference"]["chat_template"]
            s = template.get("start", "")
            if "user" in template:
                s += template["user"].replace("{text}", text)
            else:
                s += text
            s += template.get("end", "")

        return data.InferenceData(text, {})

    @torch.inference_mode()
    def _iterative_inference(
        self,
        initial_token_ids: list[int],
        entities: dict[str, dict[str, str]],
        properties: dict[str, dict[str, str]]
    ) -> Iterator[Any]:
        decoded_token_ids = []
        last_output = []

        kgs = list(self._entity_indices)
        kgs = "|".join(re.escape(kg) for kg in kgs)
        START_PATTERN = re.compile(f"<(kg(?:e|p)) kg='({kgs})'>")
        END_PATTERN = re.compile("</kg(?:e|p)>")

        index: tuple[str, str] | None = None

        def sparql_stop_fn(token_ids: list[int]) -> bool:
            nonlocal index

            if token_ids[-1] == self._eos_token_id:
                return True
            elif len(self._entity_indices) == 0:
                return False

            decoded = self.tokenizer.de_tokenize(
                token_ids[len(initial_token_ids) + len(decoded_token_ids):]
            )
            pattern = START_PATTERN if index is None else END_PATTERN
            return pattern.search(decoded) is not None

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

        initial_length = len(initial_token_ids)
        constraint = None
        is_exact = self._exact or self._force_exact

        while (
            (len(decoded_token_ids) == 0
             or decoded_token_ids[-1] != self._eos_token_id)
            and initial_length + len(decoded_token_ids) < self.max_length
        ):
            last_decoded = self.tokenizer.de_tokenize(last_output)
            if index is not None:
                match = END_PATTERN.search(last_decoded)
                assert match is not None
                name = last_decoded[:match.start()]
                assert isinstance(constraint, ContinuationConstraint)
                value = constraint.get_value()
                assert value is not None
                kg = index[1]
                if index[0] == "kge":
                    if kg not in entities:
                        entities[kg] = {}
                    entities[kg][name] = value
                else:
                    if kg not in properties:
                        properties[kg] = {}
                    properties[kg][name] = value

                index = None

            else:
                match = START_PATTERN.search(last_decoded)
                if match is not None:
                    index = (match.group(1), match.group(2))
                    if not is_exact:
                        last_output = self.tokenizer.tokenize(
                            last_decoded[:match.end()]
                        ).token_ids

            decoded_token_ids.extend(last_output)
            last_output = []

            def beam_stop_fn(beam: Beam) -> bool:
                return sparql_stop_fn(beam.token_ids)

            def token_stop_fn(token_ids: torch.Tensor, _: int) -> bool:
                return sparql_stop_fn(token_ids.tolist())

            if self._beam_width > 1 and index is not None:
                stop_fn = beam_stop_fn
                beam_width = self._beam_width
            else:
                beam_width = 1
                stop_fn = token_stop_fn

            if index is not None and index[0] == "kge":
                constraint = ContinuationConstraint(
                    self._entity_indices[index[1]]
                )
            elif index is not None and index[0] == "kgp":
                constraint = ContinuationConstraint(
                    self._property_indices[index[1]]
                )
            elif self._disable_sparql_constraint:
                constraint = None
            else:
                decoded_string = self.tokenizer.de_tokenize(decoded_token_ids)
                constraint = self._sparql_constraint
                constraint.reset(decoded_string.encode())

            i = 0
            for output, const in self._partial_inference(
                decode_fn,
                initial_token_ids + decoded_token_ids,
                constraint,
                stop_fn,  # type: ignore
                beam_width,
                kwargs_update_fn,
            ):
                yield decoded_token_ids + output
                last_output = output
                constraint = const
                i += 1

    def _partial_inference(
        self,
        decode_fn: DecodeFn,
        initial_token_ids: list[int],
        constraint: Constraint | None,
        stop_fn: BeamStopFn | StopFn,
        beam_width: int = 1,
        kwargs_update_fn: MaskUpdateFn | None = None,
    ) -> Iterator[tuple[list[int], Constraint | None]]:
        initial_length = len(initial_token_ids)
        logit_fns = []

        if beam_width > 1:
            beam = Beam(
                initial_token_ids,
                [0.0] * len(initial_token_ids),
                {"constraint": constraint}
            )

            # add constrain logit fn if any of the beams have a constraint
            if constraint is not None:
                logit_fns.append(inference_utils.constraint_logit_fn(
                    lambda beam: beam.info["constraint"],  # type: ignore
                    self._eos_token_id
                ))

                def _update_beam(beam: Beam, token_id: int, log_p: float):
                    beam = Beam.from_beam(beam, token_id, log_p)
                    beam.info["constraint"] = beam.info["constraint"].clone()
                    if token_id == self._eos_token_id:
                        return beam

                    const = beam.info["constraint"]
                    if isinstance(const, ContinuationConstraint):
                        decoded = self.tokenizer.de_tokenize(
                            beam.token_ids[initial_length:]
                        )
                        const.reset(decoded.encode())
                    else:
                        beam.info["constraint"].next(token_id)
                    return beam

                candidate_fn = _update_beam
            else:
                candidate_fn = inference_utils.default_beam_candidate_fn()

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

            for beams in beam_search(
                decode_fn=decode_fn,
                initial=[beam],
                pad_token_id=self.tokenizer.pad_token_id(),
                max_length=self.max_length,
                stop_fn=stop_fn,  # type: ignore
                device=self.devices[0],
                normalize_by_length=True,
                alpha=1.0,
                beam_width=beam_width,
                sample_fn=sample_fn,
                candidate_fn=candidate_fn,
                logit_fns=logit_fns,
                kwargs_update_fn=kwargs_update_fn,
                yield_intermediate=True
            ):
                best = beams[0][0]
                yield (best.token_ids, best.info["constraint"])
            return

        if constraint is not None:
            logit_fns.append(inference_utils.constraint_logit_fn(
                lambda _: constraint,  # type: ignore
                self._eos_token_id
            ))

        if self._sampling_strategy == "greedy":
            sample_fn = inference_utils.greedy()
        elif self._sampling_strategy == "top_k":
            logit_fns.append(inference_utils.top_k_masking(self._top_k))
            sample_fn = inference_utils.sample()
        else:
            logit_fns.append(inference_utils.nucleus_masking(self._top_p))
            sample_fn = inference_utils.sample()

        if self._sampling_strategy != "greedy" and self._temp != 1.0:
            logit_fns.append(inference_utils.temperature_scaling(
                self._temp
            ))

        # if there are constraints we need to update them
        # after sampling a token
        if constraint is not None:
            sample_fn = inference_utils.constraint_sample_fn(
                lambda _: constraint,  # type: ignore
                sample_fn,
                self._eos_token_id
            )

        yield from (
            (output[0], constraint) for output in search(
                decode_fn=decode_fn,
                initial_token_ids=[initial_token_ids],
                pad_token_id=self.tokenizer.pad_token_id(),
                max_length=self.max_length,
                sample_fn=sample_fn,
                logit_fns=logit_fns,
                stop_fn=stop_fn,  # type: ignore
                device=self.devices[0],
                kwargs_update_fn=kwargs_update_fn,
                yield_intermediate=True
            )
        )

    def set_indices(
        self,
        kg: str,
        entities: tuple[str, str] | None = None,
        properties: tuple[str, str] | None = None,
        entity_indices: tuple[ContIndex, dict[str, str]] | None = None,
        property_indices: tuple[ContIndex, dict[str, str]] | None = None,
    ) -> None:
        if kg in self._entity_indices:
            raise ValueError(f"knowledge graph {kg} already set")

        vocab = self.tokenizer.get_vocab()

        if entity_indices is not None:
            entity_index, entity_prefixes = entity_indices
        elif entities is not None:
            data, index = entities
            entity_index = ContIndex.load_with_continuations(
                os.path.join(data, "index.tsv"),
                index,
                vocab,
                common_suffix=self.cfg["inference"].get(
                    "entity_suffix",
                    "</kge>"
                )
            )
            entity_prefixes = load_prefixes(
                os.path.join(data, "prefixes.tsv")
            )
        else:
            raise ValueError("entities must be provided")

        if property_indices is not None:
            property_index, property_prefixes = property_indices
        elif properties is not None:
            data, index = properties
            property_index = ContIndex.load_with_continuations(
                os.path.join(data, "index.tsv"),
                index,
                vocab,
                common_suffix=self.cfg["inference"].get(
                    "property_suffix",
                    "</kgp>"
                )
            )
            property_prefixes = load_prefixes(
                os.path.join(data, "prefixes.tsv")
            )
        else:
            raise ValueError("properties must be provided")

        self._entity_indices[kg] = entity_index
        self._property_indices[kg] = property_index
        self._prefixes.update(entity_prefixes)
        self._prefixes.update(property_prefixes)

        # reload constraint with new kgs
        kgs = list(self._entity_indices)
        self._sparql_constraint = load_sparql_constraint(
            kgs,
            vocab,
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
        self._force_exact = force_exact

    def generate(
        self,
        query: str,
        info: str | None = None,
        examples: list[tuple[str, str]] | None = None,
        preprocessed: bool = False,
        postprocess: bool = True,
        pretty: bool = False
    ) -> Iterator[str]:
        input = self._prepare_input(query, info, examples, preprocessed)
        batch = next(self._get_loader(
            iter([input]),
            1,
        ))

        initial_token_ids = batch.token_ids()[0]
        item = batch.items()[0]

        if not self._full_outputs:
            input_text_len = len(self.tokenizer.de_tokenize(
                item.tokenization.token_ids
            ))
        else:
            input_text_len = 0

        yield input.text

        entities = {}
        properties = {}
        sparql = ""
        for output in self._iterative_inference(
            initial_token_ids,
            entities,
            properties
        ):
            sparql = self.tokenizer.de_tokenize(
                item.tokenization.token_ids + output
            )[input_text_len:]
            yield sparql

        if postprocess:
            yield postprocess_sparql_query(
                sparql,
                self._sparql_parser,
                entities,
                properties,
                self._prefixes,
                pretty
            )
