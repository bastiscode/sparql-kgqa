import logging

from text_utils.tensorboard import TensorboardMetric
from text_utils import data

import torch
from torch.utils.tensorboard import SummaryWriter


class InputOutputLogger(TensorboardMetric):
    def __init__(self, prefix: str):
        self.items = []
        self.prefix = prefix

    def set_values(self, items: list[data.TrainItem], outputs: torch.Tensor):
        self.items = items

    def get_input_output(self) -> str:
        return "\n\n".join(
            f"input:\n{item.data.input}\n\noutput:\n{item.data.target}"
            for item in self.items
        )

    def log_tensorboard(self, writer: SummaryWriter, step: int):
        writer.add_text(
            f"{self.prefix}/input_output",
            self.get_input_output(),
            step
        )

    def log_info(self, logger: logging.Logger, step: int):
        logger.info(
            f"[step {step}] {self.prefix}/input_output:\n"
            f"{self.get_input_output()}"
        )
