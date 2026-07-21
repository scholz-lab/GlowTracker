# Adopted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/shared_memory
from typing import Tuple
from dataclasses import dataclass
import multiprocessing as mp
import numpy as np

@dataclass
class ArraySpec:
    name: str
    shape: Tuple[int]
    dtype: np.dtype


class SharedAtomicCounter:
    def __init__(self, context=None):
        if context is None:
            context = mp.get_context()
        self.value = context.Value('Q', 0, lock=True)

    def load(self) -> int:
        with self.value.get_lock():
            return self.value.value

    def store(self, value: int):
        with self.value.get_lock():
            self.value.value = value

    def add(self, value: int):
        with self.value.get_lock():
            self.value.value += value
