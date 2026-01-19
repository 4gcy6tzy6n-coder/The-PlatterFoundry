"""Random Extractor - 随机种子管理"""

import random
from typing import Dict, Any


class RandomSeed:
    def __init__(self, master_seed: int):
        self.master_seed = master_seed
        self.rng = random.Random(master_seed)

    def derive_seed(self, component: str, index: int = 0) -> int:
        base = f"{self.master_seed}_{component}_{index}"
        return random.Random(base).randint(0, 2**31 - 1)

    def get_rng(self, component: str, index: int = 0) -> random.Random:
        seed = self.derive_seed(component, index)
        return random.Random(seed)


class RandomExtractor:
    def __init__(self, master_seed: int = 42):
        self.master_seed = master_seed
        self.seed = RandomSeed(master_seed)

    def extract_for_scenario(self, scenario: str, index: int = 0) -> int:
        return self.seed.derive_seed(f"scenario_{scenario}", index)

    def extract_for_algorithm(self, algorithm: str, index: int = 0) -> int:
        return self.seed.derive_seed(f"algorithm_{algorithm}", index)

    def extract_for_run(self, scenario: str, algorithm: str, repeat: int) -> int:
        return self.seed.derive_seed(f"run_{scenario}_{algorithm}", repeat)

    def get_rng(self, component: str, index: int = 0) -> random.Random:
        return self.seed.get_rng(component, index)
