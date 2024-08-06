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
from text_utils.inference.utils import Beam, ScoreFn, log_likelihood_score

from qgram_index import QGramIndex
from torch.nn.utils.rnn import pad_sequence

from sparql_kgqa.model import (
    Model,
    PretrainedDecoder,
    model_from_config,
    peft_model_from_config
)
from sparql_kgqa.sparql.utils import (
    SimilarityIndex,
    load_examples,
)
from sparql_kgqa.sparql.utils2 import (
    KgManager,
    Mapping,
    WikidataManager,
    flatten,
    partition_by,
    run_parallel,
)

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
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
        # add info to selection candidates (added automatically for duplicates)
        self._add_infos = False

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
    def _decode_fn(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        assert isinstance(self.model, PretrainedDecoder)
        dec, cache = self.model.decode(
            token_ids,
            lengths,
            kwargs.get("kv_cache", None),
            self._use_cache
        )
        return dec, {"kv_cache": cache}

    def _partial_inference(
        self,
        beams: list[Beam] | list[list[Beam]],
        stop_fn: inference_utils.StopFn,
        max_outputs: int | list[int] | None = None
    ) -> Iterator[list[list[Beam]]]:
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

        def update_fn(beam: Beam) -> Beam | None:
            # advance constraint if given
            if "const" in beam.info:
                const = beam.info["const"]
                if const.is_invalid():
                    # return None if constraint is invalid
                    # or no tokens have been generated
                    return None

                # clone constraint and update it
                const = const.clone()
                const.next(beam.token_ids[-1])
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
            decode_fn=self._decode_fn,
            initial=beams,
            pad_token_id=self.tokenizer.pad_token_id(),
            max_length=self.max_length,
            stop_fn=stop_fn,  # type: ignore
            device=self.devices[0],
            beam_width=self._beam_width,
            sample_fn=sample_fn,
            update_fn=update_fn,
            logit_fns=logit_fns,
            kwargs_update_fn=kwargs_update_fn,
            max_outputs=max_outputs,
            yield_intermediate=True
        )

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
            return beam.token_ids[-1] == self._eos_token_id

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

        # normalized log likelihood score
        score_fn = log_likelihood_score()

        outputs = [[] for _ in range(len(batch))]
        while any(beam for beam in beams):
            # run inference until eos or next entity/property
            for beams in self._partial_inference(
                beams,
                pattern_stop_fn,
                max_outputs=[
                    self._beam_width - len(output)
                    for output in outputs
                ]
            ):
                yield beams

            decoded = "\n".join(
                f"({score_fn(b):.4f}) " + b.info["decoded"]
                for b in flatten(beams)
            )
            self.logger.debug(
                "current beams:\n"
                f"{decoded}"
            )

            # filter out beams that stopped because of eos,
            # also remove duplicates
            for idx in range(len(batch)):
                stop, keep = partition_by(beams[idx], eos_stop_fn)
                outputs[idx].extend(stop)
                beams[idx] = keep

            # get continuation alternatives for non-stop beams
            beams = self._infer_alternatives(beams, eos_stop_fn)

            # rescore and keep top beams
            beams = self._rescore(beams, score_fn)
            decoded = "\n".join(
                f"({score_fn(b):.4f}) " + b.info["decoded"]
                for b in flatten(beams)
            )
            self.logger.debug(
                "beams after rescoring and pruning:\n"
                f"{decoded}"
            )

        # reorder beams
        for idx in range(len(batch)):
            outputs[idx] = sorted(
                outputs[idx],
                key=lambda beam: score_fn(beam),
                reverse=True
            )

        yield outputs

    def _rescore(
        self,
        beams: list[list[Beam]],
        score_fn: ScoreFn
    ) -> list[list[Beam]]:
        # just make a forward pass for the current beams
        # and update their log probs, sort by sequence log likelihood
        # and keep the top beams
        assert self._manager is not None, "kg indices not set"

        token_ids = []
        lengths = []
        for beam in flatten(beams):
            assert len(beam) >= 2 and beam.initial_length > 0, \
                "expected at least one input and one output token"
            token_ids.append(torch.tensor(beam.token_ids[:-1]))
            lengths.append(len(beam) - 1)

        if len(token_ids) == 0:
            # nothing to rescore
            return beams

        logits, _ = self._decode_fn(
            pad_sequence(
                token_ids,
                batch_first=True
            )[:, :self.max_length].to(self.devices[0]),
            torch.tensor(lengths, device=self.devices[0]),
        )

        log_probs = torch.log_softmax(logits, dim=-1).cpu()

        for logp, beam in zip(log_probs, flatten(beams)):
            probs = torch.gather(
                logp,
                -1,
                torch.tensor(beam.token_ids[1:]).unsqueeze(1)
            ).squeeze(1).tolist()
            assert len(probs) == len(beam) - 1
            length = beam.decoded_length
            beam.log_probs[-length:] = probs[-length:]
            assert len(beam.log_probs) == len(beam)

        for idx in range(len(beams)):
            beams[idx] = sorted(
                beams[idx],
                key=lambda beam: score_fn(beam),
                reverse=True
            )[:self._beam_width]
        return beams

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
                alts,
                max_aliases=self._max_aliases,
                add_infos=self._add_infos
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
                    "alternatives": alts,
                    "obj_type": obj_type,
                    "const": grammar.RegexConstraint(
                        regex,
                        self._continuations
                    ),
                }
            )
            select_beams.append(select_beam)

        * _, selections = self._partial_inference(
            select_beams,
            stop_fn
        )
        self.logger.debug(
            f"got {sum(len(s) for s in selections)} continuation alternatives "
            f"for {len(select_beams)} prefixes"
        )

        output_beams = [[] for _ in range(batch_size)]
        for selection in flatten(selections):
            selected = self.tokenizer.de_tokenize(selection.decoded_token_ids)
            try:
                selected = self._manager.parse_result(
                    selection.info["alternatives"],
                    selection.info["obj_type"],
                    selected
                )
            except Exception as e:
                beam = selection.info["beam"]
                decoded = beam.info["decoded"]
                self.logger.debug(
                    f"error parsing result '{selected}' "
                    f"for beam '{decoded}': {e}"
                )
                selected = None

            beam: Beam = selection.info["beam"]
            if selected is None:
                self.logger.debug(
                    f"stopping beam because none alternative was selected:\n"
                    f"{beam.info['decoded']}\n"
                    f"{beam.info['sparql']}"
                )
                continue

            identifier, name = selected
            self.logger.debug(
                f"selected {selection.info['obj_type']} alternative:\n"
                f"{beam.info['decoded']}\n"
                f"{beam.info['sparql']}\n"
                f"{beam.info['guess']} --> {name} ({identifier})"
            )
            sparql = beam.info["sparql"] + identifier
            if sparql in beam.info["seen_sparql"]:
                continue
            beam.info["seen_sparql"].add(sparql)
            # build new input from prompt and parts
            decoded = beam.info["decoded"] + name

            token_ids = self.tokenizer.tokenize(
                decoded,
                self._is_chat
            ).token_ids
            updated_beam = Beam(
                beam.initial_token_ids + token_ids,
                beam.initial_log_probs + [0.0] * len(token_ids),
                {k: v for k, v in beam.info.items()},
                beam.initial_length,
            )

            updated_beam.info["decoded"] = decoded
            updated_beam.info["sparql"] = sparql
            updated_beam.info["guess"] = ""
            updated_beam.info["last"] = len(updated_beam)
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
        select_k: int = 16,
        select_delta: int | None = None,
        select_max_candidates: int | None = 8192,
        select_max_aliases: int = 5,
        select_add_infos: bool = False,
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
        self._add_infos = select_add_infos
        self._max_aliases = select_max_aliases

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
        assert self._manager is not None, "kg indices not set"
        input = self._prepare_input(query, examples)
        batch = next(data.InferenceLoader.from_iterator(
            iter([input]),
            self.cfg["inference"]["tokenizer"],
            self.cfg["inference"].get("window", {"type": "full"}),
            ignore_special_tokens=self._is_chat
        ))

        # yield the prompt
        items = batch.items()
        yield [item.info["prompt"] for item in items]

        last: list[Beam] | None = None
        for output in self._live_inference(batch):
            beams = output[0]
            last = beams

            outputs = []
            for beam in beams:
                current = self.tokenizer.de_tokenize(
                    beam.decoded_token_ids
                )
                if pretty:
                    try:
                        current = self._manager.prettify(
                            current,
                            is_prefix=True
                        )
                    except Exception:
                        pass

                outputs.append(
                    beam.info["prompt"] + current if self._full_outputs
                    else current
                )

            yield outputs

        if not postprocess:
            return

        assert last is not None, "should not happen"

        outputs = []
        for beam in last:
            sparql = beam.info["sparql"]
            if pretty:
                try:
                    sparql = self._manager.prettify(
                        sparql,
                        is_prefix=False
                    )
                except Exception:
                    pass

            outputs.append(
                beam.info["prompt"] + sparql if self._full_outputs
                else sparql
            )

        yield outputs
