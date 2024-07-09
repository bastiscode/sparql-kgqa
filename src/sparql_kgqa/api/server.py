import time
import os
import json as J
from typing import Dict, Any

from flask import request
from flask_socketio import SocketIO, send, disconnect

from text_utils.api.server import TextProcessingServer, Error
from text_utils import continuations

from sparql_kgqa.api.generator import SPARQLGenerator
from sparql_kgqa.sparql.utils import load_prefixes


ContIndex = continuations.MmapContinuationIndex


class SPARQLGenerationServer(TextProcessingServer):
    text_processor_cls = SPARQLGenerator

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        kgs: dict[str, tuple[
            tuple[ContIndex, dict[str, str]],
            tuple[ContIndex, dict[str, str]]
        ]] = {}
        for kg, indices in config.get("knowledge_graphs", {}).items():
            assert "entity" in indices and "property" in indices, \
                f"expected 'entity' and 'property' in indices for {kg} \
                knowledge graph"

            ent_index = ContIndex.load(
                os.path.join(indices["entity"]["data"], "index.tsv"),
                indices["entity"]["index"],
                indices["entity"].get("common_suffix", "</kge>")
            )
            prop_index = ContIndex.load(
                os.path.join(indices["property"]["data"], "index.tsv"),
                indices["property"]["index"],
                indices["property"].get("common_suffix", "</kgp>")
            )
            ent_prefixes = load_prefixes(
                os.path.join(indices["entity"]["data"], "prefixes.tsv")
            )
            prop_prefixes = load_prefixes(
                os.path.join(indices["property"]["data"], "prefixes.tsv")
            )
            kgs[kg] = ((ent_index, ent_prefixes), (prop_index, prop_prefixes))

        for text_processor, model_cfg in zip(
            self.text_processors,
            config["models"]
        ):
            assert isinstance(text_processor, SPARQLGenerator)
            examples = model_cfg.get(
                "examples",
                config.get("examples", None)
            )
            example_index = model_cfg.get(
                "example_index",
                config.get("example_index", None)
            )
            if example_index is not None or examples is not None:
                text_processor.set_examples(
                    examples,
                    example_index
                )
            for kg, (entities, properties) in kgs.items():
                text_processor.set_kg_indices(
                    kg,
                    entities,
                    properties
                )

        self.socketio = SocketIO(
            self.server,
            path="live",
            cors_allowed_origins=self.allow_origin
        )

        self.connections = set()

        @self.socketio.on("connect")
        def _connect() -> None:
            self.connections.add(request.sid)  # type: ignore

        @self.socketio.on("disconnect")
        def _disconnect() -> None:
            self.connections.remove(request.sid)  # type: ignore

        @self.socketio.on("message")
        def _generate_live(data) -> None:
            try:
                json = J.loads(data)

                if "model" not in json:
                    send(J.dumps({
                        "error": "missing model in json"
                    }))
                    return
                elif "text" not in json:
                    send(J.dumps({
                        "error": "missing text in json"
                    }))
                    return

                text = json["text"]
                info = json.get("info", None)

                sampling_strategy = json.get("sampling_strategy", "greedy")
                beam_width = json.get("beam_width", 1)
                top_k = json.get("top_k", 10)
                top_p = json.get("top_p", 0.95)
                temp = json.get("temperature", 1.0)

                with self.text_processor(json["model"]) as gen:
                    if isinstance(gen, Error):
                        send(J.dumps({
                            "error": gen.msg
                        }))
                        return

                    assert isinstance(gen, SPARQLGenerator)
                    gen.set_inference_options(
                        sampling_strategy=sampling_strategy,
                        temperature=temp,
                        top_k=top_k,
                        top_p=top_p,
                        beam_width=beam_width,
                    )

                    start = time.perf_counter()
                    for text in gen.generate_live(
                        text,
                        info,
                        pretty=True
                    ):  # type: ignore
                        if request.sid not in self.connections:
                            # early explicit disconnect by client
                            return

                        send(J.dumps({
                            "output": text,
                            "runtime": {
                                "b": len(text.encode()),
                                "s": time.perf_counter() - start
                            }
                        }))

            except Exception as error:
                send(J.dumps({
                    "error": f"request failed with error: {error}"
                }))

            finally:
                disconnect()

    def run(self) -> None:
        self.socketio.run(
            self.server,
            "0.0.0.0",
            self.port,
            debug=False,
            use_reloader=False,
            log_output=False
        )
