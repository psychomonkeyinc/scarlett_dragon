import numpy as np
from typing import Tuple, Optional

class LayeredFatigue:
    """Layered fatigue model (short filename): fast synaptic + slow metabolic layers."""
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
        self.syn += self.syn_recover * dt
        if dream:
            self.meta += self.meta_rec_dream * dt
        else:
            self.meta += self.meta_rec_awake * dt
        self.syn = np.clip(self.syn, 0.01, 1.0)
        self.meta = np.clip(self.meta, self.floor, 1.0)

    def combined(self) -> np.ndarray:
        return 0.5 * self.syn + 0.5 * self.meta

    def get_vocal_efficiency(self) -> float:
        """Return vocal efficiency factor (0-1), higher is better."""
        return float(self.combined().mean())

    def apply_vocal_fatigue(self, vocal_params: np.ndarray) -> np.ndarray:
        """Apply vocal fatigue effects to vocal synthesis parameters."""
        # Simple fatigue modulation - reduce vocal energy based on overall fatigue
        fatigue_factor = self.combined().mean()
        # Apply fatigue by scaling down the parameters
        return vocal_params * fatigue_factor

    def stats(self):
        c = self.combined()
        return {
            'syn_mean': float(self.syn.mean()),
            'meta_mean': float(self.meta.mean()),
            'combined_mean': float(c.mean())
        }

# Aliases for compatibility
Fatigue = LayeredFatigue
FatigueSystem = LayeredFatigue
