from dataclasses import dataclass
from typing import Optional

from verl.workers.config.rollout import MultiTurnConfig


@dataclass
class AjetMultiTurnConfig(MultiTurnConfig):

    max_sample_per_task: int = 30
    max_steps: int = 30
    expected_steps: Optional[int] = None