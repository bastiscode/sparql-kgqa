import os
from typing import Dict, Any, Tuple

import torch
from torch import nn

from text_utils.api.trainer import ShardingPolicy, Trainer
from text_utils import data

from sparql_kgqa.model import (
    model_from_config,
    peft_model_from_config
)


class TextGenerationTrainer(Trainer):
    @classmethod
    def _model_from_config(
        cls,
        cfg: Dict[str, Any]
    ) -> Tuple[nn.Module, ShardingPolicy | None]:
        model = model_from_config(cfg["model"])
        if cfg.get("gradient_checkpointing", False):
            model.enable_gradient_checkpointing()
        return model, model.get_sharding_policy()

    @classmethod
    def _prepare_peft(
        cls,
        model: nn.Module,
        cfg: dict[str, Any],
    ) -> nn.Module:
        return peft_model_from_config(
            model,  # type: ignore
            cfg
        )

    def _prepare_batch(self, batch: data.TrainBatch) -> Tuple[
        Dict[str, Any],
        torch.Tensor
    ]:
        assert len(batch) > 0, "got empty batch"

        inputs = batch.tensors()
        input_type = inputs.pop("type")
        assert input_type == "generation", \
            f"unexpected input type: {input_type}"

        labels = torch.from_numpy(
            inputs.pop("labels")
        ).to(
            non_blocking=True,
            dtype=torch.long,
            device=self.info.device
        )
        inputs = {
            k: torch.from_numpy(v).to(
                non_blocking=True,
                dtype=torch.int,
                device=self.info.device
            )
            for k, v in inputs.items()
        }
        return inputs, labels


def main():
    parser = TextGenerationTrainer.parser(
        "Text generation", "Train a model for generating text"
    )
    args = parser.parse_args()
    work_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
    if args.platform == "local":
        TextGenerationTrainer.train_local(
            work_dir, args.experiment, args.config, args.profile
        )
    else:
        TextGenerationTrainer.train_slurm(
            work_dir, args.experiment, args.config
        )


if __name__ == "__main__":
    main()
