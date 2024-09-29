from typing import Any, Generator, Iterable, Iterator
import logging
import random

from search_index.index import SearchIndex
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
from text_utils.constraints import AndConstraint, AvoidConstraint

from sparql_kgqa.model import (
    Model,
    PretrainedDecoder,
    model_from_config,
    peft_model_from_config
)
from sparql_kgqa.sparql.utils import load_examples
from sparql_kgqa.sim_index import SimilarityIndex
from sparql_kgqa.sparql.utils2 import (
    KgManager,
    Mapping,
    WikidataManager,
    Chat,
    get_kg_manager,
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
        self._max_length = None
        self._max_new_tokens = None

        # search index options
        # number of alternatives to select from
        self._k = 8
        # maximum size where sub index is built
        self._max_candidates: int | None = 4096
        # add info to selection candidates (added automatically for duplicates)
        self._add_infos = False

        # SPARQL stuff
        self._exact = self.cfg["inference"].get("exact", False)
        self._force_exact = False
        self._sparql_constraint: None | grammar.LR1Constraint = None
        self._disable_subgraph_constraint = False
        self._disable_sparql_constraint = False
        self._disable_sparql_judgement = False
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
        self._max_failures: int = 3
        self._min_tries: int | None = None
        self._max_alternatives: int = 3
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
            examples = self._example_index.find_matches(
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

        messages = self._manager.get_sparql_continuation_prompt(
            question,
            "",
            examples
        )
        prompt = self._chat_format(messages)
        return data.InferenceData(
            prompt,
            {
                "question": question,
                "prompt": prompt,
            }
        )

    def _chat_format(
        self,
        input: str | Chat,
    ) -> str:
        if isinstance(input, str):
            input = [{"role": "user", "text": input}]

        assert "chat_template" in self.cfg["inference"], \
            "chat template not set in config, expect one even for " \
            "non-chat models (use default)"
        chat_template = self.cfg["inference"]["chat_template"]

        assert "user" in chat_template["roles"], \
            "chat template must have a user role"
        assert "assistant" in chat_template["roles"], \
            "chat template must have an assistant role"

        s: str = chat_template.get("start", "")

        system_message = self._system_message or self._default_system_message
        if system_message is not None:
            assert all(m["role"] != "system" for m in input), \
                "system message already in input"
            assert "system" in chat_template["roles"], \
                "chat template must have a system role"
            s += chat_template["roles"]["system"].replace(
                "{text}",
                system_message
            )

        last_partial = False
        for i, message in enumerate(input):
            role = message["role"]
            text = message["text"]
            assert role in chat_template["roles"], \
                f"role {role} not in chat template"
            template = chat_template["roles"][role]
            if message.get("partial", False):
                assert i == len(input) - 1, "partial message not last"
                pos = template.find("{text}")
                s += template[:pos] + text
                last_partial = True
            else:
                s += template.replace("{text}", text)

        if not last_partial:
            s += chat_template.get("end", "")

        return s

    @torch.inference_mode()
    def _decode_fn(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        assert isinstance(self.model, PretrainedDecoder)
        dec, _ = self.model.decode(
            token_ids,
            lengths,
        )
        return dec, {}

    def _partial_inference(
        self,
        beam: Beam,
        stop_fn: inference_utils.StopFn,
        no_sampling: bool = False
    ) -> Iterator[Beam]:
        def update_fn(beam: Beam) -> Beam | None:
            const = beam.info.get("const", None)
            if const is None:
                return beam
            elif const.is_invalid():
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

        if self._sampling_strategy == "greedy" or no_sampling:
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

        for beams in beam_search(
            decode_fn=self._decode_fn,
            initial=[beam],
            pad_token_id=self.tokenizer.pad_token_id(),
            max_length=self.max_length,
            stop_fn=stop_fn,  # type: ignore
            device=self.devices[0],
            beam_width=self._beam_width,
            sample_fn=sample_fn,
            update_fn=update_fn,
            logit_fns=logit_fns,
            max_new_tokens=self._max_new_tokens,
            yield_intermediate=True
        ):
            beam = beams[0][0]
            yield beam

        yield beam

    def _live_inference(
        self,
        batch: data.InferenceBatch
    ) -> Iterator[str]:
        assert self._manager is not None, "kg indices not set"
        assert self._sparql_constraint is not None, "sparql constraint not set"
        # decode fn gets in token ids and additional kwargs,
        # and return logits over next tokens and additional info

        self._sparql_constraint.reset()

        batch_size = len(batch)
        assert batch_size == 1, \
            "only supporting single batched inference for now"

        info = batch.infos()[0]
        token_ids = batch.token_ids()[0]
        question = info["question"]

        beam = Beam(token_ids)
        if not self._disable_sparql_constraint:
            beam.info["const"] = self._sparql_constraint.clone()

        memo: dict[tuple[tuple[str, str], ...], list] = {}
        current: list = []
        states: list[str] = ["sparql", "search", "select"]

        def advance(value):
            current.append(value)

        def backtrack():
            assert current, "cannot backtrack"
            value = current.pop()
            key = tuple(current)
            if key in memo:
                memo[key].append(value)
            else:
                memo[key] = [value]

        def failures() -> list:
            return memo.get(tuple(current), [])

        def num_failures() -> int:
            return len(failures())

        def prefix(typ: str) -> str:
            assert self._manager is not None, "kg indices not set"
            assert typ in {"natural", "sparql"}
            pfx = ""
            for i, val in enumerate(current):
                s = state(i)
                if s == "sparql":
                    pfx += " " * (i > 0) + val[0]
                elif s == "select" and typ == "natural":
                    pfx += val[0]
                elif s == "select" and typ == "sparql":
                    obj_type, identifier, variant = val[1]
                    iri = self._manager.denormalize_selection(
                        obj_type,
                        identifier,
                        variant
                    )
                    pfx += iri or identifier
                # skip search, done and none states
            return pfx

        def state(idx: int | None = None) -> str | None:
            # translate current to state name,
            # can be one of:
            # - sparql (continuing query ending in search token)
            # - done (same as sparql but with empty search token)
            # - search (generating search query)
            # - select (selecting alternative)

            # always return current state if no specific idx
            # is given
            if idx is None:
                idx = len(current)

            name = states[idx % len(states)]
            if name != "search" or idx == 0:
                return name

            # last one was sparql, if sparql produced empty
            # search token (finished sparql query) we are done
            _, search = previous(idx - 1)
            return "done" if search == "" else name

        def previous(idx: int | None = None) -> Any:
            assert current, "no previous state"
            return current[idx if idx is not None else -1]

        # init state
        s = state()
        assert s == "sparql", "initial state must be sparql"

        while s:
            prev = state(len(current) - 1) if current else 'start'
            self.logger.debug(
                f"current state:\n"
                f"name:     {s}\n"
                f"sparql:   {prefix('sparql')}\n"
                f"natural:  {prefix('natural')}\n"
                f"failures: {failures()}\n"
                f"previous: {prev} "
                f" {previous() if current else 'none'}"
            )

            num = num_failures()
            # go back if too many failures
            if num > self._max_failures:
                if current:
                    backtrack()
                    s = state()
                    continue

                # cannot backtrack, return empty sparql
                break

            # force different path if not
            # enough were already tried
            exclude_failures = num < (self._min_tries or num)

            if s == "done":
                accept = self._judge_sparql(
                    question,
                    prefix("sparql"),
                    prefix("natural")
                )
                if accept:
                    # breaking early will return final sparql query
                    break

                # continue with search
                backtrack()

            elif s == "sparql":
                # continue with sparql query
                failed = set(
                    continuation + search_token
                    for continuation, search_token in failures()
                )
                continuation, search_token = yield from self._continue_sparql(
                    question,
                    prefix("sparql"),
                    prefix("natural"),
                    failed,
                    exclude_failures
                )
                if continuation + search_token in failed:
                    if current:
                        backtrack()
                    else:
                        # cannot backtrack, return empty sparql
                        break
                else:
                    advance((continuation, search_token))

            elif s == "search":
                failed = set(failures())
                search_query = self._generate_search_query(
                    question,
                    prefix("natural"),
                    failed,
                    exclude_failures
                )
                if search_query in failed:
                    backtrack()
                else:
                    advance(search_query)

            else:
                search_query = previous()
                failed = set(selection for _, selection in failures())
                selection = self._select_alternative(
                    question,
                    prefix("sparql"),
                    prefix("natural"),
                    search_query,
                    failed,
                    exclude_failures
                )
                if selection is None or selection[1] in failed:
                    backtrack()
                else:
                    advance(selection)

            # update state
            s = state()

        yield prefix("sparql")

    def _judge_sparql(
        self,
        question: str,
        sparql: str,
        natural_sparql: str
    ) -> bool:
        if self._disable_sparql_judgement:
            return True
        assert self._manager is not None, "kg indices not set"
        prompt, regex = self._manager.get_judgement_prompt_and_regex(
            question,
            sparql,
            natural_sparql,
        )
        token_ids = self.tokenizer.tokenize(
            self._chat_format(prompt),
            self._is_chat
        ).token_ids
        beam = Beam(
            token_ids,
            info={
                "const": grammar.RegexConstraint(
                    regex,
                    self._continuations
                ),
            }
        )

        *_, beam = self._partial_inference(
            beam,
            lambda beam: beam.token_ids[-1] == self._eos_token_id,
        )

        judgement = self.tokenizer.de_tokenize(beam.decoded_token_ids)
        explanation, answer = self._manager.parse_judgement(judgement)
        self.logger.debug(f"judgement:\n{prompt}{judgement}")
        self.logger.debug(f"answer: {answer}, explanation: {explanation}")
        return answer

    def _continue_sparql(
        self,
        question: str,
        sparql_prefix: str,
        natural_prefix: str,
        failures: set[str] | None = None,
        exclude_failures: bool = False
    ) -> Generator[str, None, tuple[str, str]]:
        assert self._manager is not None, "kg indices not set"
        assert self._sparql_constraint is not None, "sparql constraint not set"
        prompt = self._manager.get_sparql_continuation_prompt(
            question,
            natural_prefix,
            failures=failures
        )
        token_ids = self.tokenizer.tokenize(
            self._chat_format(prompt),
            self._is_chat
        ).token_ids

        info: dict = {"continuation": "", "search_token": ""}
        # add sparql constraint if not disabled
        if not self._disable_sparql_constraint:
            const = self._sparql_constraint.clone()
            const.reset(sparql_prefix.encode())
            info["const"] = const

        if exclude_failures and failures:
            const = AvoidConstraint(
                [f.encode() for f in failures],
                self._continuations,
                self._eos_token_id
            )
            if "const" in info:
                info["const"] = AndConstraint([
                    info["const"],
                    const
                ])
            else:
                info["const"] = const

        beam = Beam(
            token_ids,
            info=info
        )

        def eos_stop_fn(beam: Beam) -> bool:
            return beam.token_ids[-1] == self._eos_token_id

        def stop_fn(beam: Beam) -> bool:
            decoded = self.tokenizer.de_tokenize(beam.decoded_token_ids)
            beam.info["continuation"] = decoded
            assert self._manager is not None, "kg indices not set"
            if eos_stop_fn(beam):
                return True

            match = self._manager.search_pattern.search(decoded)
            if match is None:
                return False

            part = decoded[:match.start()]
            beam.info["continuation"] = part
            beam.info["search_token"] = match.group(0)
            return True

        last: Beam | None = None
        for output in self._partial_inference(
            beam,
            stop_fn,
            no_sampling=True  # for sparql continuations turn off sampling
        ):
            self.logger.debug(self.tokenizer.de_tokenize(
                output.decoded_token_ids
            ))
            yield self.tokenizer.de_tokenize(output.decoded_token_ids)
            last = output

        assert last is not None, "should not happen"
        cont = last.info["continuation"]
        search_token = last.info["search_token"]
        self.logger.debug(
            "continuation:\n"
            f"{prompt[-2]['text']}"
            f"{(prompt[-1]['text'] + ' ' + cont + search_token).lstrip()}"
        )
        return cont, search_token

    def _generate_search_query(
        self,
        question: str,
        prefix: str,
        failures: set[str] | None = None,
        exclude_failures: bool = False
    ) -> str:
        assert self._manager is not None, "kg indices not set"
        prompt, regex = self._manager.get_search_prompt_and_regex(
            question,
            prefix,
            failures
        )
        token_ids = self.tokenizer.tokenize(
            self._chat_format(prompt),
            self._is_chat
        ).token_ids
        beam = Beam(
            token_ids,
            info={
                "const": AndConstraint([
                    grammar.RegexConstraint(
                        regex,
                        self._continuations
                    ),
                    AvoidConstraint(
                        [f.encode() for f in failures]
                        if exclude_failures and failures is not None
                        else [],
                        self._continuations,
                        self._eos_token_id
                    )
                ])
            }
        )

        *_, search = self._partial_inference(
            beam,
            lambda beam: beam.token_ids[-1] == self._eos_token_id
        )

        search_query = self.tokenizer.de_tokenize(search.decoded_token_ids)
        self.logger.debug(f"search:\n{prompt}{search_query}")
        return search_query

    def _select_alternative(
        self,
        question: str,
        sparql_prefix: str,
        natural_prefix: str,
        search_query: str,
        failures: set[tuple[str, str, str | None]] | None = None,
        exclude_failures: bool = False
    ) -> tuple[str, tuple[str, str, str | None]] | None:
        assert self._manager is not None, "kg indices not set"

        alternatives = self._manager.get_selection_alternatives(
            sparql_prefix,
            search_query,
            self._k,
            self._max_candidates,
            max_retries=3,
            skip_autocomplete=self._disable_subgraph_constraint
        )
        if alternatives is None:
            return None

        prompt, regex = self._manager.get_selection_prompt_and_regex(
            question,
            natural_prefix,
            search_query,
            alternatives,
            max_aliases=self._max_aliases,
            add_infos=self._add_infos,
            failures=failures,
            exclude_failures=exclude_failures
        )
        token_ids = self.tokenizer.tokenize(
            self._chat_format(prompt),
            self._is_chat
        ).token_ids
        beam = Beam(
            token_ids,
            info={
                "const": grammar.RegexConstraint(
                    regex,
                    self._continuations
                ),
            }
        )

        *_, selection = self._partial_inference(
            beam,
            lambda beam: beam.token_ids[-1] == self._eos_token_id
        )

        selected = self.tokenizer.de_tokenize(selection.decoded_token_ids)
        self.logger.debug(f"selection:\n{prompt}{selected}")
        return self._manager.parse_selection(
            alternatives,
            selected
        )

    def set_examples(
        self,
        examples: Examples | str | None = None,
        example_index: SimilarityIndex | str | list[str] | None = None
    ) -> None:
        self._example_index = None
        self._examples = None
        if isinstance(example_index, SimilarityIndex):
            self._example_index = example_index

        elif (
            isinstance(example_index, str)
            or isinstance(example_index, list)
        ):
            index = SimilarityIndex()
            index.load(example_index)
            self._example_index = index

        elif isinstance(examples, str):
            self._examples = load_examples(examples)

        elif isinstance(examples, list):
            self._examples = examples

        else:
            raise ValueError("either examples or example_index must be set")

    def set_kg_indices(
        self,
        kg: str,
        entity_index: SearchIndex,
        property_index: SearchIndex,
        entity_mapping: Mapping,
        property_mapping: Mapping
    ) -> None:
        self._manager = get_kg_manager(
            kg,
            entity_index,
            property_index,
            entity_mapping,
            property_mapping
        )
        self._sparql_constraint = self._manager.get_constraint(
            self._continuations,
            self._exact or self._force_exact
        )

    def set_inference_options(
        self,
        sampling_strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.95,
        beam_width: int = 1,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        disable_sparql_constraint: bool = False,
        disable_subgraph_constraint: bool = False,
        disable_sparql_judgement: bool = False,
        max_failures: int = 3,
        min_tries: int = 1,
        num_examples: int = 3,
        select_k: int = 8,
        select_max_candidates: int | None = 4096,
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
        self._max_new_tokens = max_new_tokens
        self._disable_sparql_constraint = disable_sparql_constraint
        self._disable_subgraph_constraint = disable_subgraph_constraint
        self._disable_sparql_judgement = disable_sparql_judgement
        self._force_exact = force_exact
        assert max_failures >= (min_tries or max_failures), \
            f"got max {max_failures} failures but min {min_tries} tries"
        self._num_examples = num_examples
        self._max_failures = max_failures
        self._min_tries = min_tries
        self._system_message = system_message
        self._k = select_k
        self._max_candidates = select_max_candidates
        self._add_infos = select_add_infos
        self._max_aliases = select_max_aliases

    def generate(
        self,
        inputs: Iterable[tuple[str, Examples | None]],
        sort: bool = True,
        num_threads: int | None = None,
        show_progress: bool = False,
        pretty: bool = False
    ) -> Iterator[str | list[str]]:
        def inference_fn(batch: data.InferenceBatch) -> list[str]:
            *_, last = self._live_inference(batch)
            return [last]

        def postprocessing_fn(
            items: list[data.InferenceItem],
            outputs: list[str],
        ) -> str | list[str]:
            assert len(items) == 1 and len(outputs) == 1
            assert self._manager is not None, "kg indices not set"
            output = outputs[0]

            if pretty:
                output = self._manager.prettify(
                    output,
                    is_prefix=True
                )

            return output

        yield from self._process(
            (self._prepare_input(*input) for input in inputs),
            inference_fn,
            postprocessing_fn,
            "Generating SPARQL queries",
            batch_size=1,
            sort=sort,
            num_threads=num_threads,
            show_progress=show_progress,
            ignore_special_tokens=self._is_chat
        )

    def generate_live(
        self,
        query: str,
        examples: Examples | None = None,
        postprocess: bool = True,
        pretty: bool = False
    ) -> Iterator[str]:
        assert self._manager is not None, "kg indices not set"
        input = self._prepare_input(query, examples)
        batch = next(data.InferenceLoader.from_iterator(
            iter([input]),
            self.cfg["inference"]["tokenizer"],
            self.cfg["inference"].get("window", {"type": "full"}),
            ignore_special_tokens=self._is_chat
        ))

        # yield the prompt
        item = batch.items()[0]
        yield item.info["prompt"]

        output = ""
        for output in self._live_inference(batch):
            yield output

        if not postprocess:
            return

        if pretty:
            try:
                output = self._manager.prettify(
                    output,
                    is_prefix=False
                )
            except Exception:
                pass

        yield output
