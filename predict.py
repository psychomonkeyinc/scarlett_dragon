# predict.py

import numpy as np
import logging
import pickle  # COMMENTED OUT: persistence logic
from typing import Dict, List, Optional, Tuple, Any

# Assume nn.py is available for Sequential, Linear, Sigmoid, ReLU, Tanh
from nn import Sequential, Linear, Sigmoid, ReLU, Tanh, mse_loss_prime
from config import (PREDICT_CAFVE_TOKEN_DIM, PREDICT_PREDICTOR_DIM, 
                   PREDICT_SFE_FEATURE_DIM, PREDICT_UNIFIED_COGNITIVE_STATE_DIM,
                   PREDICT_EMOTIONAL_STATE_DIM, PREDICT_OTHER_MIND_MODEL_DIM,
                   PREDICT_FOCUS_VECTOR_DIM, PREDICT_HIDDEN_DIM)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Predict:
    """
    Lillith's Active Predictive Module.
    Selectively predicts high-level events (next sensory input, internal state, other agent behavior)
    to drive learning through prediction error.
    """
    def __init__(self,
                 sfe_feature_dim: int,
                 cafve_token_dim: int,
                 unified_cognitive_state_dim: int,
                 emotional_state_dim: int,
                 other_mind_model_dim: int,
                 focus_vector_dim: int,
                 predicted_output_dim: int,
                 prediction_error_dim: int):
        
        self.sfe_feature_dim = sfe_feature_dim
        self.cafve_token_dim = cafve_token_dim
        self.unified_cognitive_state_dim = unified_cognitive_state_dim
        self.emotional_state_dim = emotional_state_dim
        self.other_mind_model_dim = other_mind_model_dim
        self.focus_vector_dim = focus_vector_dim
        self.predicted_output_dim = predicted_output_dim
        self.prediction_error_dim = prediction_error_dim

        self.prediction_input_dim = (self.sfe_feature_dim + 
                                     self.cafve_token_dim + 
                                     self.unified_cognitive_state_dim + 
                                     self.other_mind_model_dim +
                                     self.focus_vector_dim +
                                     self.emotional_state_dim)

        self.predictor_network = Sequential(
            Linear(self.prediction_input_dim, PREDICT_PREDICTOR_DIM),
            ReLU(),
            Linear(PREDICT_PREDICTOR_DIM, self.predicted_output_dim)
        )
        
        self._last_prediction = np.zeros(self.predicted_output_dim, dtype=np.float32)
        self._last_error = np.zeros(self.prediction_error_dim, dtype=np.float32)

        logger.info(f"Predict module initialized. Input Dim: {self.prediction_input_dim}, Predicted Output Dim: {self.predicted_output_dim}")

    def predict(self, 
                sfe_features: np.ndarray, 
                cafve_tokens: List[np.ndarray],
                unified_cognitive_state: np.ndarray, 
                other_mind_model: np.ndarray,
                focus_vector: np.ndarray,
                emotional_state: np.ndarray) -> np.ndarray:
        """Generates predictions based on current context."""
        avg_cafve_token = np.mean(np.array(cafve_tokens), axis=0).astype(np.float32) if cafve_tokens else np.zeros(self.cafve_token_dim, dtype=np.float32)

        combined_input = np.concatenate([
            sfe_features,
            avg_cafve_token,
            unified_cognitive_state.astype(np.float32),
            other_mind_model,
            focus_vector,
            emotional_state
        ]).astype(np.float32).reshape(1, -1)

        if combined_input.shape[1] != self.prediction_input_dim:
            logger.error(f"Predict: Input dim mismatch. Expected {self.prediction_input_dim}, got {combined_input.shape[1]}. Returning last prediction.")
            return self._last_prediction.reshape(1, -1)

        self._last_prediction = self.predictor_network.forward(combined_input)
        return self._last_prediction

    def update_and_learn_predictions(self, 
                                     sfe_features: np.ndarray, 
                                     cafve_token_current: np.ndarray,
                                     unified_cognitive_state: np.ndarray,
                                     other_mind_model: np.ndarray, 
                                     current_focus_vector: np.ndarray, 
                                     emotional_state: np.ndarray, 
                                     optimizer: Any):
        """Generates a new prediction and learns from the error of the *previous* prediction."""
        # Generate new prediction for the *next* cycle
        self.predict(sfe_features, [cafve_token_current], unified_cognitive_state, 
                     other_mind_model, current_focus_vector, emotional_state)
        
        # Calculate error: compare last cycle's prediction with current actual state
        self._last_error = unified_cognitive_state.astype(np.float32) - self._last_prediction[0, :]
        
        # Learn from this error (backpropagate through the predictor network)
        loss_grad = mse_loss_prime(unified_cognitive_state.astype(np.float32), self._last_prediction)
        self.predictor_network.backward(loss_grad)
        # The main loop will call optimizer.step()

    def get_prediction_error(self) -> np.ndarray:
        return self._last_error.copy()

    def get_prediction_error_norm(self) -> float:
        return float(np.linalg.norm(self._last_error))

    def get_predicted_next_state(self) -> np.ndarray:
        return self._last_prediction.copy()

    def predict_next(self, 
                     sfe_features: np.ndarray, 
                     cafve_tokens: List[np.ndarray],
                     unified_cognitive_state: np.ndarray, 
                     other_mind_model: np.ndarray,
                     focus_vector: np.ndarray,
                     emotional_state: np.ndarray) -> np.ndarray:
        """Generates predictions for the next cognitive state."""
        return self.predict(sfe_features, cafve_tokens, unified_cognitive_state, 
                           other_mind_model, focus_vector, emotional_state)

    def get_networks(self) -> None:
        return [self.predictor_network]

    # def save_state(self, save_path: str):
    #     try:
    #         state = {
    #             'last_prediction': self._last_prediction.tolist(),  # COMMENTED OUT
    #             'last_error': self._last_error.tolist(),  # COMMENTED OUT
    #             'predictor_network_weights': [(p[0].tolist(), p[1]) for p in self.predictor_network.get_trainable_params()]  # COMMENTED OUT
    #         }
    #         with open(save_path, 'wb') as f:
    #             # pickle.dump(state, f)  # COMMENTED OUT
    #             pass
    #         logger.info(f"Predict state saved to {save_path}")
    #     except Exception as e:
    #         logger.error(f"Error saving Predict state: {e}")

    # def load_state(self, load_path: str):
    #     try:
    #         with open(load_path, 'rb') as f:
    #             # state = pickle.load(f)  # COMMENTED OUT
    #             pass
    #         # self._last_prediction = np.array(state['last_prediction'], dtype=np.float32)  # COMMENTED OUT
    #         # self._last_error = np.array(state['last_error'], dtype=np.float32)  # COMMENTED OUT
    #         # loaded_params = state.get('predictor_network_weights', [])  # COMMENTED OUT
    #         # current_params = self.predictor_network.get_trainable_params()  # COMMENTED OUT
    #         # if len(loaded_params) == len(current_params):  # COMMENTED OUT
    #         #     for i, (param_val_list, grad_name_str) in enumerate(loaded_params):  # COMMENTED OUT
    #         #         param_array, _, layer_instance = current_params[i]  # COMMENTED OUT
    #         #         param_array[:] = np.array(param_val_list, dtype=np.float32)  # COMMENTED OUT
    #         # else: logger.warning("Predictor network weights mismatch.")  # COMMENTED OUT
    #         # logger.info(f"Predict state loaded from {load_path}")  # COMMENTED OUT
    #     except FileNotFoundError:
    #         logger.warning(f"Predict state file not found at {load_path}.")
    #         pass
    #     except Exception as e:
    #         logger.error(f"Error loading Predict state: {e}.")
    #         pass

 
