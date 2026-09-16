from typing import Any


class ProgressLogger:
    def __init__(self, logger: Any):
        self.logger = logger

    def on_step(self, metrics: dict[str, float]) -> None:
        self.logger.info("Training step metrics: %s", metrics)

    def on_epoch_end(self, metrics: dict[str, float]) -> None:
        self.logger.info("Epoch end metrics: %s", metrics)
