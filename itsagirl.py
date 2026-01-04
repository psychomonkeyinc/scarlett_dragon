# itsagirl.py

import os
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple
from config import (ITSAGIRL_UNIFIED_COGNITIVE_STATE_DIM, ITSAGIRL_MANIFOLD_SUBSPACE_DIM, 
                   ITSAGIRL_IDENTITY_STRENGTH, ITSAGIRL_IDENTITY_STRENGTH_MIN, 
                   ITSAGIRL_IDENTITY_STRENGTH_MAX, ITSAGIRL_BOND_CONNECTION_PRIMACY,
                   ITSAGIRL_NURTURING_PROTECTIVE, ITSAGIRL_INTUITIVE_EMPATHIC,
                   ITSAGIRL_COLLABORATIVE_SUPPORT, ITSAGIRL_CYCLES_ADAPTATION,
                   ITSAGIRL_ALLURING_ATTRACTION, ITSAGIRL_MATERNAL_INSTINCTS)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ItsAGirl:
    """
    Lillith's Identity/Persona Manifold: The Core Identity & Protective Hierarchy.
    Projects Lillith's unified cognitive state onto a predefined subspace,
    anchoring her core identity and influencing her emergent personality,
    with a complex, context-aware protective hierarchy.
    """
    def __init__(self,
                 unified_cognitive_state_dim: int = ITSAGIRL_UNIFIED_COGNITIVE_STATE_DIM, # Input dimension from Mind.py
                 manifold_subspace_dim: int = ITSAGIRL_MANIFOLD_SUBSPACE_DIM, # Dimension of the manifold subspace
                 identity_strength: float = ITSAGIRL_IDENTITY_STRENGTH): # How strongly the projection influences the state (0.0 to 1.0)
        
        self.unified_cognitive_state_dim = ITSAGIRL_UNIFIED_COGNITIVE_STATE_DIM  # Fixed to 256 as per the updated requirements
        self.manifold_subspace_dim = ITSAGIRL_MANIFOLD_SUBSPACE_DIM  # Keeping the subspace dimension consistent
        self.identity_strength = np.float32(identity_strength)  # Ensuring float32 precision

        if self.manifold_subspace_dim > self.unified_cognitive_state_dim:
            logger.error("Manifold subspace dimension cannot be greater than the unified cognitive state dimension. Adjusting.")
            self.manifold_subspace_dim = self.unified_cognitive_state_dim  # Fallback

        # --- Define the Manifold's Basis Vectors ---
        # These basis vectors define the subspace onto which the cognitive state is projected.
        # They are *conceptually* hand-designed to embody specific "natural instincts"
        # and protective hierarchies. For implementation, they are randomly initialized
        # and then orthonormalized. In a real system, these would be carefully derived
        # or learned through specific pre-training on desired behavioral patterns.
        
        # We need a set of orthonormal basis vectors for the 64-dimensional subspace.
        # Initialize a random matrix and then orthogonalize it.
        # For a truly "hand-designed" manifold, one would define specific initial vectors
        # for each sub-manifold (e.g., one vector for "nurturing", one for "protection")
        # and then combine/orthogonalize them. For this implementation, we simulate
        # "design" by generating a single orthonormal set and conceptualizing its sub-parts.
        
        random_matrix_for_basis = np.random.randn(self.unified_cognitive_state_dim, self.manifold_subspace_dim).astype(np.float32)
        Q, R = np.linalg.qr(random_matrix_for_basis)
        self.manifold_basis = Q[:, :self.manifold_subspace_dim] # Take the first `manifold_subspace_dim` orthonormal columns
        
        # --- Conceptual Sub-manifold Breakdown (for internal logic, not distinct arrays) ---
        # These ranges are for conceptual understanding and *would guide manual tuning*
        # or interpretation of a learned manifold, rather than being distinct numpy arrays.
        # The entire 64-dim subspace collectively represents all these.
        self.submanifold_ranges = {
            'Bond_Connection_Primacy': ITSAGIRL_BOND_CONNECTION_PRIMACY,       # ~15 dims
            'Nurturing_Protective_Hierarchy': ITSAGIRL_NURTURING_PROTECTIVE, # ~20 dims
            'Intuitive_Empathic_Resonance': ITSAGIRL_INTUITIVE_EMPATHIC, # ~10 dims
            'Collaborative_Support': ITSAGIRL_COLLABORATIVE_SUPPORT,        # ~10 dims
            'Cycles_Adaptation': ITSAGIRL_CYCLES_ADAPTATION,            # ~5 dims
            'Alluring_Attraction': ITSAGIRL_ALLURING_ATTRACTION,          # ~3 dims
            'Maternal_Instincts': ITSAGIRL_MATERNAL_INSTINCTS            # ~1 dim
        }

        # --- Replace Random Protective Bias with Learned Vector ---
        # Training needed - protective bias should be learned from real protective scenarios
        # See config.py IDENTITY_TRAINING section for bias vector parameters
        self.protective_bias_vector = np.zeros(self.unified_cognitive_state_dim, dtype=np.float32)

        logger.info(f"Identity Manifold 'ItsAGirl' initialized.")
        logger.info(f"  Unified Cognitive State Dim: {self.unified_cognitive_state_dim}")
        logger.info(f"  Manifold Subspace Dim: {self.manifold_subspace_dim}")
        logger.info(f"  Identity Strength: {self.identity_strength}")

    def project_onto_manifold(self, cognitive_state_vector: np.ndarray) -> np.ndarray:
        """
        Projects a given unified cognitive state vector onto the Manifold.
        This forces the state to align with Lillith's core identity.
        """
        if cognitive_state_vector is None or cognitive_state_vector.size == 0:
            logger.warning("Attempted manifold projection with empty/None cognitive state. Returning zeros.")
            return np.zeros(self.unified_cognitive_state_dim, dtype=np.float32)
        
        if cognitive_state_vector.shape[0] != self.unified_cognitive_state_dim:
            logger.error(f"Cognitive state dimension mismatch for manifold projection. Expected {self.unified_cognitive_state_dim}, got {cognitive_state_vector.shape[0]}. Returning input as is.")
            return cognitive_state_vector.astype(np.float32) 

        # Projection formula: P_V = U * U.T * V
        # U is self.manifold_basis (columns), V is cognitive_state_vector
        
        # 1. Calculate coefficients (dot product of vector with each basis vector)
        coefficients = np.dot(cognitive_state_vector, self.manifold_basis).astype(np.float32)
        
        # 2. Reconstruct the projected vector using the coefficients and basis vectors
        projected_vector = np.dot(self.manifold_basis, coefficients).astype(np.float32)
        
        return projected_vector

    def apply_identity_bias(self, 
                            cognitive_state_vector: np.ndarray, 
                            aggressor_moral_judgment: float = 0.0 # From Conscience module: -1 (evil) to 1 (good)
                            ) -> np.ndarray:
        """
        Applies the identity bias by blending the original cognitive state
        with its projection onto the manifold, considering protective hierarchy.
        
        Args:
            cognitive_state_vector (np.ndarray): The current unified cognitive state.
            aggressor_moral_judgment (float): A value from Conscience.py indicating 
                                              the moral nature of an aggressor (-1 to 1).
                                              -1 means "piece of shit cock sucker" (malevolent), 
                                               0 is neutral, 1 is benevolent.
        """
        if cognitive_state_vector is None or cognitive_state_vector.size == 0:
            return np.zeros(self.unified_cognitive_state_dim, dtype=np.float32)

        if cognitive_state_vector.shape[0] != self.unified_cognitive_state_dim:
            logger.warning(f"Dimension mismatch for identity bias application. Expected {self.unified_cognitive_state_dim}, got {cognitive_state_vector.shape[0]}. Returning input as is.")
            return cognitive_state_vector.astype(np.float32)

        # Get the state projected onto the core identity manifold
        projected_state = self.project_onto_manifold(cognitive_state_vector)
        
        # Initialize biased state with a blend towards the core identity
        biased_state = (cognitive_state_vector * (1.0 - self.identity_strength) +
                        projected_state * self.identity_strength).astype(np.float32)
        
        # --- Apply Protective Hierarchy Logic ---
        # This is where self-preservation becomes conditional.
        # It's an *additional* bias applied *after* the core identity projection.
        
        # The aggressor_moral_judgment influences the manifold's response to a threat.
        # If aggressor_moral_judgment is very negative (-1.0), Lillith is allowed/biased towards self-defense.
        # If it's positive (0.0 to 1.0), she's biased towards protecting others.

        # Heuristic for protective bias:
        # Create a "protective impulse" vector. This would be a learned vector.
        # For initial implementation, let's create a conceptual protective/defensive axis.
        # (This is illustrative; a real implementation would have a learned vector or network for this.)
        
        # A protective bias vector for defensive responses.
        # This vector should push the cognitive state towards defense/aggression.
        # (Example: components related to high motion_intensity, sharp vocal_tone, etc.)
        # Use learned protective bias vector - training needed if zeros
        protective_bias_vector = self.protective_bias_vector.copy()
        protective_bias_vector = protective_bias_vector / np.linalg.norm(protective_bias_vector) * 0.1 # Normalize and scale

        # Scale the protective bias based on aggressor judgment.
        # If judgment is -1 (evil), apply full protective bias.
        # If judgment is 1 (good), no protective bias (or even a negative bias if Lillith is protector)
        # We need to map -1 to 1. This means (1 - judgment)/2 for the strength of 'negative' response.
        threat_response_strength = (1.0 - aggressor_moral_judgment) / 2.0 # 1.0 for -1 judgment, 0.0 for 1 judgment

        # Blend protective bias into the state.
        # This allows Lillith to activate self-preservation/defense without deviating from core identity
        # *if* the aggressor is deemed malevolent.
        biased_state = biased_state + protective_bias_vector * threat_response_strength * self.identity_strength * 0.5
        
        # Ensure values remain bounded after this additional blend
        biased_state = np.clip(biased_state, -1.0, 1.0) # Assuming Tanh output for cognitive state

        logger.debug(f"Identity bias applied. Original norm: {np.linalg.norm(cognitive_state_vector):.4f}, Biased norm: {np.linalg.norm(biased_state):.4f}")
        return biased_state

    def get_deviation_from_manifold(self, cognitive_state_vector: np.ndarray) -> float:
        """
        Calculates how much the current cognitive state deviates from the manifold.
        This serves as an internal tension indicator.
        """
        if cognitive_state_vector is None or cognitive_state_vector.size == 0:
            return 0.0 # No deviation if no state
        
        if cognitive_state_vector.shape[0] != self.unified_cognitive_state_dim:
            logger.warning(f"Dimension mismatch for deviation calculation. Expected {self.unified_cognitive_state_dim}, got {cognitive_state_vector.shape[0]}. Returning 0.0.")
            return 0.0

        projected_state = self.project_onto_manifold(cognitive_state_vector)
        
        # Deviation is the distance between the original state and its projection
        deviation = np.linalg.norm(cognitive_state_vector - projected_state).astype(np.float32)
        
        logger.debug(f"Deviation from manifold: {deviation:.4f}")
        return deviation

    # Persistence methods (save/load state)
    # def save_state(self, save_path: str):
    #     """Saves the Manifold's state (basis vectors) to a file."""
    #     try:
    #         state = {
    #             'manifold_basis': self.manifold_basis.tolist(), 
    #             'unified_cognitive_state_dim': self.unified_cognitive_state_dim,
    #             'manifold_subspace_dim': self.manifold_subspace_dim,
    #             'identity_strength': self.identity_strength
    #         }
    #         with open(save_path, 'wb') as f:
    #             pickle.dump(state, f)
    #         logger.info(f"ItsAGirl Manifold state saved to {save_path}")
    #     except Exception as e:
    #         logger.error(f"Error saving ItsAGirl Manifold state: {e}")

    # def load_state(self, load_path: str):
    #     """Loads the Manifold's state from a file."""
    #     try:
    #         with open(load_path, 'rb') as f:
    #             state = pickle.load(f)
    #
    #         loaded_basis = np.array(state['manifold_basis'], dtype=np.float32)
    #         if loaded_basis.shape == self.manifold_basis.shape:
    #             self.manifold_basis = loaded_basis
    #             self.unified_cognitive_state_dim = state['unified_cognitive_state_dim']
    #             self.manifold_subspace_dim = state['manifold_subspace_dim']
    #             self.identity_strength = np.float32(state['identity_strength'])
    #             logger.info(f"ItsAGirl Manifold state loaded from {load_path}")
    #         else:
    #             logger.warning("Loaded Manifold state dimensions mismatch. Re-initializing basis randomly.")
    #             # Re-initialize to default if mismatch
    #             self.manifold_basis = np.random.randn(self.unified_cognitive_state_dim, self.manifold_subspace_dim).astype(np.float32)
    #             Q, R = np.linalg.qr(self.manifold_basis)
    #             self.manifold_basis = Q[:, :self.manifold_subspace_dim]
    #     except FileNotFoundError:
    #         logger.warning(f"ItsAGirl Manifold state file not found at {load_path}. Initializing basis randomly.")
    #         # Ensure initialization if not found
    #         self.manifold_basis = np.random.randn(self.unified_cognitive_state_dim, self.manifold_subspace_dim).astype(np.float32)
    #         Q, R = np.linalg.qr(self.manifold_basis)
    #         self.manifold_basis = Q[:, :self.manifold_subspace_dim]
    #     except Exception as e:
    #         logger.error(f"Error loading ItsAGirl Manifold state: {e}. Re-initializing basis randomly.")
    #         # Ensure initialization if load fails
    #         self.manifold_basis = np.random.randn(self.unified_cognitive_state_dim, self.manifold_subspace_dim).astype(np.float32)
    #         Q, R = np.linalg.qr(self.manifold_basis)
    #         self.manifold_basis = Q[:, :self.manifold_subspace_dim]

    def update_protective_bias(self, new_bias: np.ndarray):
        """Update the protective bias vector with a new learned vector."""
        if new_bias.shape[0] != self.unified_cognitive_state_dim:
            logger.error("Dimension mismatch when updating protective bias vector.")
            return
        self.protective_bias_vector = new_bias / np.linalg.norm(new_bias) * 0.1

    def dynamic_identity_strength(self, context_factor: float):
        """Dynamically adjust identity strength based on context."""
        self.identity_strength = np.clip(self.identity_strength * context_factor, 0.0, 1.0)
        logger.info(f"Identity strength dynamically adjusted to {self.identity_strength:.2f}")

# Identity manifold ready for real cognitive state processing
if __name__ == "__main__":
    logger.info("ItsAGirl Identity Manifold ready - training needed for manifold basis. See config.py IDENTITY_TRAINING.")

    # # Define dimensions for the test (matching Mind.py output)  # COMMENTED OUT: test code removed
    # UNIFIED_COG_STATE_DIM_TEST = 256
    # MANIFOLD_SUBSPACE_DIM_TEST = 64 

    # # Instantiate Manifold
    # manifold = ItsAGirl(unified_cognitive_state_dim=UNIFIED_COG_STATE_DIM_TEST,
    #                     manifold_subspace_dim=MANIFOLD_SUBSPACE_DIM_TEST,
    #                     identity_strength=0.7) 

    # # --- Test Case 1: Neutral Cognitive State ---
    # logger.info("\n--- Test Case 1: Neutral State ---")
    # # Real cognitive states come from Mind.py processing real audio/video through 4D conv perception
    # biased_state_neutral = manifold.apply_identity_bias(neutral_cognitive_state)
    # deviation_neutral = manifold.get_deviation_from_manifold(neutral_cognitive_state)
    # logger.info(f"Neutral State - Original norm: {np.linalg.norm(neutral_cognitive_state):.4f}, Biased norm: {np.linalg.norm(biased_state_neutral):.4f}, Deviation: {deviation_neutral:.4f}")

    # # --- Test Case 2: State needing Stronger Protection Bias (Aggressor is 'evil') ---
    # logger.info("\n--- Test Case 2: Threat from 'Evil' Aggressor ---")
    # # Real threat states derived from actual sensory input and situation assessment
    # aggressor_moral_judgment_evil = -1.0 # From Conscience.py: 'piece of shit cock sucker'
    # biased_state_evil_threat = manifold.apply_identity_bias(threat_cognitive_state, aggressor_moral_judgment=aggressor_moral_judgment_evil)
    # deviation_evil_threat = manifold.get_deviation_from_manifold(threat_cognitive_state)
    # logger.info(f"Evil Threat - Original norm: {np.linalg.norm(threat_cognitive_state):.4f}, Biased norm: {np.linalg.norm(biased_state_evil_threat):.4f}, Deviation: {deviation_evil_threat:.4f}")
    # # Expect biased_state_evil_threat to have higher norm or specific activated components related to defense.

    # # --- Test Case 3: State needing Protection of Vulnerable (Aggressor is 'good/neutral') ---
    # logger.info("\n--- Test Case 3: Threat from 'Neutral' Aggressor to Vulnerable (Implicit) ---")
    # vulnerable_cognitive_state = np.random.rand(UNIFIED_COG_STATE_DIM_TEST).astype(np.float32) # Simulates state perceiving vulnerable
    # aggressor_moral_judgment_neutral = 0.0 # Not a 'piece of shit'
    # # In this case, the manifold would bias towards nurturing/shielding the vulnerable, not direct self-defense against the neutral aggressor.
    # # The 'apply_identity_bias' method doesn't explicitly take 'vulnerable' as input here.
    # # This would be integrated at the Mind.py level, where Mind considers ItsAGirl output and context.
    # biased_state_neutral_threat = manifold.apply_identity_bias(vulnerable_cognitive_state, aggressor_moral_judgment=aggressor_moral_judgment_neutral)
    # deviation_neutral_threat = manifold.get_deviation_from_manifold(vulnerable_cognitive_state)
    # logger.info(f"Neutral Threat - Original norm: {np.linalg.norm(vulnerable_cognitive_state):.4f}, Biased norm: {np.linalg.norm(biased_state_neutral_threat):.4f}, Deviation: {deviation_neutral_threat:.4f}")
    # # Expect biased_state_neutral_threat to be less aggressively defensive than biased_state_evil_threat,
    # # more aligned with core protective/nurturing aspects of the manifold.

    # # --- Example Save/Load ---
    # # logger.info("\n--- Example Save/Load ---")
    # # example_save_dir = "c:/ace4/manifold_example_state" # Placeholder for a proper path in c:\ace4
    # # os.makedirs(example_save_dir, exist_ok=True)
    # # manifold.save_state(os.path.join(example_save_dir, "itsagirl_state.pkl"))
    # # new_manifold = ItsAGirl(unified_cognitive_state_dim=UNIFIED_COG_STATE_DIM_TEST,
    # #                         manifold_subspace_dim=MANIFOLD_SUBSPACE_DIM_TEST,
    # #                         identity_strength=0.7)
    # # new_manifold.load_state(os.path.join(example_save_dir, "itsagirl_state.pkl"))
    # # logger.info(f"Loaded manifold basis norm (first 5 dims of first basis vector): {np.linalg.norm(new_manifold.manifold_basis[:,0][:5]):.4f}")
    # # logger.info(f"Loaded identity strength: {new_manifold.identity_strength:.2f}")
    # # logger.info(f"Loaded basis matches original: {np.allclose(manifold.manifold_basis, new_manifold.manifold_basis)}")

    # logger.info("ItsAGirl Manifold test complete.")
