"""Lightweight logger (console + optional TensorBoard)."""
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str = None, use_tb: bool = True):
        self.writer = SummaryWriter(log_dir) if use_tb and log_dir else None
        self._step = 0

    def log(self, tag: str, value: float, step: int = None):
        step = step if step is not None else self._step
        print(f"[{step:6d}] {tag}: {value:.4f}")
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def step(self):
        self._step += 1

    def close(self):
        if self.writer:
            self.writer.close()
