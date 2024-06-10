import time
import json as J
from typing import Dict, Any

from flask_sock import Sock

from text_utils.api.server import TextProcessingServer, Error

from sparql_kgqa.api.generator import SPARQLGenerator


class SPARQLGenerationServer(TextProcessingServer):
    text_processor_cls = SPARQLGenerator

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        use_cache = config.get("use_cache", False)

        kgs = {}
        for kg, indices in config.get("knowledge_graphs", {}).items():
            assert "entity" in indices and "property" in indices, \
                f"expected 'entity' and 'property' in indices for {kg} \
                knowledge graph"
            kgs[kg] = (indices["entity"], indices["property"])

        for cfg, text_processor in zip(config["models"], self.text_processors):
            for kg in cfg.get("knowledge_graphs", []):
                assert isinstance(text_processor, SPARQLGenerator)
                ent_index, prop_index = kgs[kg]
                text_processor.set_indices(ent_index, prop_index, kg)

        self.websocket = Sock(self.server)

        @self.websocket.route(f"{self.base_url}/live")
        def _generate_live(ws) -> None:
            try:
                data = ws.receive(timeout=self.timeout)
                json = J.loads(data)
                if json is None:
                    ws.send(J.dumps({
                        "error": "request body must be json"
                    }))
                    return
                elif "model" not in json:
                    ws.send(J.dumps({
                        "error": "missing model in json"
                    }))
                    return
                elif "text" not in json:
                    ws.send(J.dumps({
                        "error": "missing text in json"
                    }))
                    return

                text = json["text"]
                info = json.get("info", None)

                sampling_strategy = json.get("sampling_strategy", "greedy")
                beam_width = json.get("beam_width", None)
                top_k = json.get("top_k", 10)
                top_p = json.get("top_p", 0.95)
                temp = json.get("temperature", 1.0)
                max_length = json.get("max_length", None)

                with self.text_processor(json["model"]) as gen:
                    if isinstance(gen, Error):
                        ws.send(J.dumps({
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
                        use_cache=use_cache,
                        max_length=max_length
                    )

                    start = time.perf_counter()
                    for text in gen.generate_live(text, info):  # type: ignore
                        ws.send(J.dumps({
                            "output": text,
                            "runtime": {
                                "b": len(text.encode()),
                                "s": time.perf_counter() - start
                            }
                        }))

            except Exception as error:
                ws.send(J.dumps({
                    "error": f"request failed with unexpected error: {error}"
                }))
            finally:
                ws.close()