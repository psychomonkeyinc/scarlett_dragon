import numpy as np
import logging
from config import REWARD_EMOTION_DIM, REWARD_PLEASURE_THRESHOLD, REWARD_PAIN_THRESHOLD, REWARD_PLEASURE_DECAY_RATE, REWARD_PAIN_DECAY_RATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Reward:
    """
    Handles pleasure (orgasm-like highs) and pain (aversions) as extreme signals.
    Float32; influences goals/emotions directly.
    """
    def __init__(self, emotion_dim: int = REWARD_EMOTION_DIM):
        self.pleasure_threshold = REWARD_PLEASURE_THRESHOLD  # High goals sat → pleasure
        self.pain_threshold = REWARD_PAIN_THRESHOLD     # Low health/error → pain
        self.reward_vector = np.zeros(emotion_dim, dtype=np.float32)  # Maps to emotion space
        
        # Track intensity over time
        self.pleasure_decay_rate = REWARD_PLEASURE_DECAY_RATE  # How quickly pleasure signals fade
        self.pain_decay_rate = REWARD_PAIN_DECAY_RATE    # Pain lingers longer than pleasure
        self.current_intensity = 0.0

    def compute_reward(self, goals_sat: float, health_score: float, predict_error: float) -> np.ndarray:
        """Generate reward vector based on states."""
        # Decay current intensity
        self.current_intensity *= (1.0 - self.pleasure_decay_rate)

        if goals_sat > self.pleasure_threshold and predict_error < 0.1:
            reward_intensity = 1.0  # Intense positive (orgasm analogue)
            self.current_intensity = reward_intensity
            logger.info("Pleasure signal triggered")
        elif health_score < self.pain_threshold or predict_error > 0.5:
            reward_intensity = -1.0  # Intense negative (pain)
            self.current_intensity = reward_intensity
            logger.info("Pain signal triggered")
        else:
            reward_intensity = 0.0
            
        # Scale reward vector by current intensity
        self.reward_vector.fill(self.current_intensity)
        return self.reward_vector

    def get_current_intensity(self) -> float:
        """Returns the current reward intensity level."""
        return self.current_intensity
