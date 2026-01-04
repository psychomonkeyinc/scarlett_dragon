import numpy as np
from typing import Tuple, Optional

class LayeredFatigue:
    """Layered fatigue model: fast synaptic + slow metabolic components.
    Provides combined fatigue map plus update & activation hooks.
    """
    def __init__(self,
                 shape: Tuple[int, int],
                 syn_recover: float = 0.03,
                 syn_cost: float = 0.5,
                 meta_decay: float = 0.002,
                 meta_rec_awake: float = 0.0004,
                 meta_rec_dream: float = 0.01,
                 floor: float = 0.05):
        self.syn = np.full(shape, 1.0, dtype=np.float32)
        self.meta = np.full(shape, 1.0, dtype=np.float32)
        self.syn_recover = np.float32(syn_recover)
        self.syn_cost = np.float32(syn_cost)
        self.meta_decay = np.float32(meta_decay)
        self.meta_rec_awake = np.float32(meta_rec_awake)
        self.meta_rec_dream = np.float32(meta_rec_dream)
        self.floor = np.float32(floor)

    def apply_activation(self, coord: Optional[Tuple[int,int]], neighborhood_mean: float):
        if coord is None:
            return
        self.syn[coord] -= self.syn_cost
        self.meta[coord] -= self.meta_decay * (0.5 + neighborhood_mean)

    def recover(self, dt: float, dream: bool):
        # Fast layer recovery
        self.syn += self.syn_recover * dt
        # Slow layer recovery (faster while dreaming)
        if dream:
            self.meta += self.meta_rec_dream * dt
        else:
            self.meta += self.meta_rec_awake * dt
        # Clamp
        self.syn = np.clip(self.syn, 0.01, 1.0)
        self.meta = np.clip(self.meta, self.floor, 1.0)

    def combined(self) -> np.ndarray:
        return 0.5 * self.syn + 0.5 * self.meta

    def stats(self):
        return {
            'syn_mean': float(self.syn.mean()),
            'meta_mean': float(self.meta.mean()),
            'combined_mean': float(self.combined().mean())
        }
