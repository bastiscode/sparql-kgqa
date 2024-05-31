from typing import Any, Iterator

import torch
from torch import nn
from peft import get_peft_model

from text_utils import data, tokenization, grammar
from text_utils.api.processor import ModelInfo, TextProcessor
from text_utils.api.utils import (
    Device,
    device_info,
    get_devices,
    get_peft_config
)
from text_utils.inference import (
    utils as inference_utils,
    search,
    beam_search
)
from text_utils.inference.utils import Beam
from text_utils.constraints import Constraint

from sparql_kgqa.model import (
    Model,
    PretrainedDecoder,
    model_from_config
)
from sparql_kgqa.sparql.utils import (
    ContIndex,
    load_sparql_constraint,
    preprocess_natural_language_query
)

_BASE_URL = ""
_NAME_TO_ZIP = {}

Const = str | tuple[str, str, bool]


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
        _: Device
    ) -> nn.Module:
        model = model_from_config(cfg["model"])
        peft = cfg["train"].get("peft", None)
        if peft is not None:
            peft_cfg = get_peft_config(peft)
            model.model = get_peft_model(
                model.model,  # type: ignore
                peft_cfg
            )
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
        self._sparql_constraint = load_sparql_constraint(
            [],
            self.tokenizer.get_vocab(),
            False
        )

        self.model = self.model.compile(
            **self.cfg["inference"].get("compile", {})
        )

        self._entity_indices = {}
        self._property_indices = {}

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
        preprocessed: bool = False
    ) -> data.InferenceData:
        if not preprocessed:
            text = preprocess_natural_language_query(
                text,
                list(self._entity_indices),
                info
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
    def _inference(
        self,
        batch: data.InferenceBatch,
        constraint: Constraint | None = None,
        yield_intermediate: bool = False
    ) -> Iterator[Any]:
        initial_token_ids = batch.token_ids()
        if constraint is not None:
            constraint.reset()

        # decode fn gets in token ids and additional kwargs,
        # and return logits over next tokens and additional info
        def _decode_fn(
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

        def _kwargs_update_fn(
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

        if (self._beam_width or 1) > 1:
            assert self._beam_width is not None
            logit_fns = []
            initial_beams = []

            for token_ids in initial_token_ids:
                beam = Beam(token_ids, [0.0] * len(token_ids))

                if constraint is not None:
                    beam.info["constraint"] = constraint.clone()

                initial_beams.append(beam)

            # add constrain logit fn if any of the beams have a constraint
            if constraint is not None:
                logit_fns.append(inference_utils.constraint_logit_fn(
                    lambda beam: beam.info["constraint"],  # type: ignore
                    self._eos_token_id
                ))

                def _update_beam(beam: Beam, token_id: int, log_p: float):
                    new_beam = Beam.from_beam(beam, token_id, log_p)
                    if token_id == self._eos_token_id:
                        return new_beam

                    beam_const = beam.info["constraint"].clone()
                    beam_const.next(token_id)
                    new_beam.info["constraint"] = beam_const
                    return new_beam

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

            def beam_stop_fn(beam: Beam) -> bool:
                return beam.token_ids[-1] == self._eos_token_id

            yield from (
                [beam[0].token_ids for beam in beams]
                for beams in
                beam_search(
                    decode_fn=_decode_fn,
                    initial=initial_beams,
                    pad_token_id=self.tokenizer.pad_token_id(),
                    max_length=self.max_length,
                    stop_fn=beam_stop_fn,
                    device=self.devices[0],
                    normalize_by_length=True,
                    alpha=1.0,
                    beam_width=self._beam_width,
                    sample_fn=sample_fn,
                    candidate_fn=candidate_fn,
                    logit_fns=logit_fns,
                    kwargs_update_fn=_kwargs_update_fn,
                    yield_intermediate=yield_intermediate
                )
            )
            return

        logit_fns = []

        if constraint is not None:
            constraints = [
                constraint.clone()
                for _ in range(len(initial_token_ids))
            ]
            logit_fns.append(inference_utils.constraint_logit_fn(
                lambda idx: constraints[idx],  # type: ignore
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
                lambda idx: constraints[idx],  # type: ignore
                sample_fn,
                self._eos_token_id
            )

        def stop_fn(token_ids: torch.Tensor, _: list[int]) -> torch.Tensor:
            return token_ids == self._eos_token_id

        yield from search(
            decode_fn=_decode_fn,
            initial_token_ids=initial_token_ids,
            pad_token_id=self.tokenizer.pad_token_id(),
            max_length=self.max_length,
            sample_fn=sample_fn,
            logit_fns=logit_fns,
            stop_fn=stop_fn,
            device=self.devices[0],
            kwargs_update_fn=_kwargs_update_fn,
            yield_intermediate=yield_intermediate
        )

    def _process_results(
        self,
        items: list[data.InferenceItem],
        outputs: list[Any],
    ) -> data.InferenceData:
        assert len(outputs) == 1, "expected single output"
        output = outputs[0]
        item = items[0]

        text = self.tokenizer.de_tokenize(
            item.tokenization.token_ids + output
        )
        if not self._full_outputs:
            input_text = self.tokenizer.de_tokenize(
                item.tokenization.token_ids
            )
            text = text[len(input_text):]
        return data.InferenceData(text, item.data.info)

    def _get_constraint(
        self,
        constraint: Const
    ) -> Constraint:
        if isinstance(constraint, str):
            return grammar.RegexConstraint(
                constraint,
                self._continuations
            )
        else:
            gram, lexer, exact = constraint
            return grammar.LR1Constraint(
                gram,
                lexer,
                self._continuations,
                exact
            )

    def set_indices(
        self,
        entity_index: str | ContIndex,  # type: ignore
        property_index: str | ContIndex,  # type: ignore
        kg: str
    ) -> None:
        if kg in self._entity_indices:
            raise ValueError(f"knowledge graph {kg} already set")

        vocab = self.tokenizer.get_vocab()
        if isinstance(entity_index, str):
            entity_index = ContIndex.load_with_continuations(
                entity_index,
                vocab
            )
        if isinstance(property_index, str):
            property_index = ContIndex.load_with_continuations(
                property_index,
                vocab
            )

        self._entity_indices[kg] = entity_index
        self._property_indices[kg] = property_index
        # reload constraint with new kgs
        self._sparql_constraint = load_sparql_constraint(
            list(self._entity_indices),
            vocab,
            False
        )

    def set_inference_options(
        self,
        sampling_strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.95,
        beam_width: int | None = None,
        max_length: int | None = None,
        use_cache: bool = False,
        full_outputs: bool = False
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

    def generate_live(
        self,
        text: str,
        info: str | None = None,
        preprocessed: bool = False
    ) -> Iterator[str]:
        batch = next(self._get_loader(
            iter([self._prepare_input(text, info, preprocessed)]),
            1,
        ))
        items = None
        for outputs in self._inference(batch, yield_intermediate=True):
            if items is None:
                items = batch.items()
            yield self._process_results(items, outputs).text
