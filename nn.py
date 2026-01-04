# nn.py

import numpy as np
from config import FLOATX, NN_HE_INITIALIZATION_SCALE_BASE, NN_PLASTICITY_RATE, NN_PLASTICITY_DECAY, NN_FATIGUE_RATE, NN_RECOVERY_RATE, NN_FATIGUE_THRESHOLD, NN_TARGET_ACTIVITY, NN_SCALING_RATE, NN_FATIGUE_MASK_MAX, NN_FATIGUE_MASK_MIN, NN_FATIGUE_ACTIVITY_THRESHOLD, NN_ACTIVITY_HISTORY_DECAY, NN_ACTIVITY_HISTORY_UPDATE, NN_PLASTIC_CLEANUP_FACTOR
from typing import List, Tuple, Any, Dict, Optional
import logging
import time
from OptiJustinJ import JustinJOptimizer, PerformanceMonitor   

# ==============================================================================
# SECTION 1: THE NEURAL FOUNDATION
# A from-scratch, pure NumPy replacement for a neural network library,
# including explicit backpropagation for custom optimization.
# ==============================================================================

class Layer:
    """
    The abstract base class for all neural network layers.
    Defines the required forward and backward pass methods.
    """
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Calculates the output of the layer for a given input.
        """
        raise NotImplementedError

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient to be passed to the previous layer and
        updates internal parameters (weights/biases) if they exist.
        """
        raise NotImplementedError

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """Allows layers to be called directly like a function."""
        return self.forward(input_data)

    def get_trainable_params(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Returns a list of (parameter, gradient) tuples for this layer."""
        return []

    def get_state(self) -> Dict[str, Any]:
        """Returns the serializable state of the layer."""
        return {}

    def set_state(self, state: Dict[str, Any]):
        """Restores the state of the layer."""
        pass

class Linear(Layer):
    """
    A dense, fully connected neural layer with biological properties.
    This replacement keeps the original behavior but stores the primary
    arrays in float16 as requested (note: some ops may upcast temporarily).
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        # He initialization: sqrt(scale_base / fan_in)
        scale_base = getattr(__import__('config'), 'NN_HE_INITIALIZATION_SCALE_BASE', 2.0)
        scale = np.sqrt(scale_base / float(input_size)).astype(FLOATX) if isinstance(input_size, int) else FLOATX(np.sqrt(scale_base / float(input_size)))
        self.weights = (np.random.randn(input_size, output_size).astype(FLOATX) * scale)
        self.biases = np.zeros((1, output_size), dtype=FLOATX)

        # Plasticity factors
        self.plasticity_rate = FLOATX(NN_PLASTICITY_RATE)
        self.plasticity_decay = FLOATX(NN_PLASTICITY_DECAY)
        self.plastic_changes = np.zeros_like(self.weights, dtype=FLOATX)

        # Neural fatigue
        self.fatigue = np.zeros(output_size, dtype=FLOATX)
        self.fatigue_rate = FLOATX(NN_FATIGUE_RATE)
        self.recovery_rate = FLOATX(NN_RECOVERY_RATE)
        self.fatigue_threshold = FLOATX(NN_FATIGUE_THRESHOLD)

        # Homeostatic scaling
        self.target_activity = FLOATX(NN_TARGET_ACTIVITY)
        self.scaling_rate = FLOATX(NN_SCALING_RATE)
        self.activity_history = np.ones(output_size, dtype=FLOATX) * FLOATX(NN_TARGET_ACTIVITY)

        # Memory bounds to prevent leaks
        self.max_plastic_magnitude = FLOATX(1.0)
        self.plastic_cleanup_threshold = FLOATX(0.001)

        
        self.weights_gradient = (None)
        self.biases_gradient = (None)

        # Training mode flag
        self.training = (True)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        # Accept inputs, store as float32
        self.input = input_data.astype(FLOATX)

        # Linear transform
        raw_output = np.dot(self.input, self.weights) + self.biases

        # Fatigue modulation
        fatigue_mask_max = getattr(__import__('config'), 'NN_FATIGUE_MASK_MAX', 1.0)
        fatigue_mask_min = getattr(__import__('config'), 'NN_FATIGUE_MASK_MIN', 0.0)
        fatigue_mask = FLOATX(fatigue_mask_max) - np.clip(self.fatigue - self.fatigue_threshold, FLOATX(fatigue_mask_min), FLOATX(fatigue_mask_max))
        modulated_output = raw_output * fatigue_mask

        # Update fatigue
        fatigue_activity_threshold = getattr(__import__('config'), 'NN_FATIGUE_ACTIVITY_THRESHOLD', 0.1)
        self.fatigue += self.fatigue_rate * (np.abs(modulated_output) > FLOATX(fatigue_activity_threshold)).mean(axis=0)
        self.fatigue *= (FLOATX(fatigue_mask_max) - self.recovery_rate)

        # Hebbian plasticity (simple correlation-based)
        if self.training:
            correlation = np.dot(self.input.T, modulated_output) / max(1, self.input.shape[0])
            self.plastic_changes = (self.plastic_changes * self.plasticity_decay) + (correlation * self.plasticity_rate)
            self.plastic_changes = np.clip(self.plastic_changes, -self.max_plastic_magnitude, self.max_plastic_magnitude)
            small_changes_mask = np.abs(self.plastic_changes) < self.plastic_cleanup_threshold
            plastic_cleanup_factor = getattr(__import__('config'), 'NN_PLASTIC_CLEANUP_FACTOR', 0.1)
            self.plastic_changes[small_changes_mask] *= FLOATX(plastic_cleanup_factor)
            effective_weights = self.weights + self.plastic_changes
        else:
            effective_weights = self.weights

        # Final output
        self.output = np.dot(self.input, effective_weights) + self.biases

        # Activity history and homeostatic scaling
        current_activity = (self.output > 0).mean(axis=0).astype(FLOATX)
        activity_history_decay = getattr(__import__('config'), 'NN_ACTIVITY_HISTORY_DECAY', 0.9)
        activity_history_update = getattr(__import__('config'), 'NN_ACTIVITY_HISTORY_UPDATE', 0.1)
        self.activity_history = (self.activity_history * FLOATX(activity_history_decay)) + (current_activity * FLOATX(activity_history_update))
        activity_scale = (FLOATX(1.0) + (self.scaling_rate * (self.target_activity - self.activity_history)))
        self.output = (self.output * activity_scale).astype(FLOATX)

        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Scale gradient based on fatigue (upcast for numerical ops where needed)
        fatigue_mask = (FLOATX(1.0) - np.clip(self.fatigue - self.fatigue_threshold, FLOATX(0.0), FLOATX(1.0))).astype(FLOATX)
        scaled_gradient = (output_gradient * fatigue_mask).astype(FLOATX)

        effective_weights = (self.weights + self.plastic_changes).astype(FLOATX)
        self.weights_gradient = np.dot(self.input.T, scaled_gradient)
        self.biases_gradient = np.sum(scaled_gradient, axis=0, keepdims=True)

        activity_error = (self.activity_history - self.target_activity).astype(FLOATX)
        homeostatic_gradient = (self.scaling_rate * activity_error * np.sign(self.output)).astype(FLOATX)
        if self.weights_gradient.shape[0] == 1 and homeostatic_gradient.ndim == 1:
            self.weights_gradient += homeostatic_gradient.reshape(1, -1)
        else:
            try:
                self.weights_gradient += homeostatic_gradient.reshape(1, -1)
            except Exception:
                pass

        input_gradient = np.dot(scaled_gradient, effective_weights.T).astype(FLOATX)
        return input_gradient

    def get_trainable_params(self) -> List[Tuple[np.ndarray, Any]]:
        """Returns list of (parameter_array, gradient_array) tuples. Training needs to happen for gradients to exist."""
        return [(self.weights, 'weights_gradient'), (self.biases, 'biases_gradient')]

    def train(self):
        """Set the layer to training mode."""
        self.training = True

    def eval(self):
        """Set the layer to evaluation mode."""
        self.training = False

class Tanh(Layer):
    """A hyperbolic tangent activation function."""
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = np.tanh(self.input)
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Derivative of tanh(x) is 1 - tanh(x)^2
        return output_gradient * (1 - self.output**2)

class Sigmoid(Layer):
    """A sigmoid activation function, squashing values between 0 and 1."""
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = np.clip(input_data, -500, 500)  # Clip inputs to prevent overflow
        # Numerically stable sigmoid
        self.output = np.where(
            self.input >= 0,
            1 / (1 + np.exp(-self.input)),
            np.exp(self.input) / (1 + np.exp(self.input))
        )
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
        return output_gradient * (self.output * (1 - self.output))

class ReLU(Layer):
    """A Rectified Linear Unit activation function."""
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = np.maximum(0, self.input)
        return self.output
        
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Derivative of ReLU is 1 for positive inputs and 0 otherwise.
        relu_mask = (self.input > 0).astype(np.float32)
        return output_gradient * relu_mask

class GELU(Layer):
    """A Gaussian Error Linear Unit, a smoother version of ReLU."""
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        # Standard GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        self.output = 0.5 * self.input * (1 + np.tanh(np.sqrt(2 / np.pi) * (self.input + 0.044715 * self.input**3)))
        return self.output
        
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Approximate derivative for the GELU function
        x = self.input
        tanh_arg = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        tanh_val = np.tanh(tanh_arg)
        sech_squared_val = 1 - tanh_val**2 
        derivative_inner_term = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
        
        derivative = 0.5 * (1 + tanh_val) + 0.5 * x * sech_squared_val * derivative_inner_term
        
        return output_gradient * derivative

class LayerNorm(Layer):
    """
    Applies Layer Normalization over the last dimension of the input.
    Helps stabilize learning in deep networks.
    """
    def __init__(self, features: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        # Gamma (scale) and Beta (shift) are learnable parameters
        self.gamma = np.ones((1, features), dtype=FLOATX)
        self.beta = np.zeros((1, features), dtype=FLOATX)

        # Gradients for gamma and beta
        self.gamma_gradient = None
        self.beta_gradient = None

        # Internal variables saved for backward pass
        self.mean = None
        self.variance = None
        self.normalized_input = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data.astype(FLOATX)
        
        # Calculate mean and variance over the last dimension (features)
        self.mean = np.mean(self.input, axis=-1, keepdims=True)
        self.variance = np.var(self.input, axis=-1, keepdims=True)
        
        # Normalize the input
        self.normalized_input = (self.input - self.mean) / np.sqrt(self.variance + self.epsilon)
        
        # Apply learnable scaling (gamma) and shifting (beta)
        self.output = self.gamma * self.normalized_input + self.beta
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Gradients for beta and gamma
        self.beta_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        self.gamma_gradient = np.sum(output_gradient * self.normalized_input, axis=0, keepdims=True)

        # Gradient with respect to normalized input (dL/d(normalized_input))
        d_normalized_input = output_gradient * self.gamma

        # Gradient with respect to variance (dL/d(variance))
        d_variance = np.sum(d_normalized_input * (self.input - self.mean) * -0.5 * np.power(self.variance + self.epsilon, -1.5), axis=-1, keepdims=True)

        # Gradient with respect to mean (dL/d(mean))
        d_mean = np.sum(d_normalized_input * (-1 / np.sqrt(self.variance + self.epsilon)), axis=-1, keepdims=True) + \
                 d_variance * (-2 * (self.input - self.mean)).mean(axis=-1, keepdims=True)

        # Gradient with respect to input data (dL/d(input_data))
        input_gradient = d_normalized_input * (1 / np.sqrt(self.variance + self.epsilon)) + \
                         d_variance * (2 * (self.input - self.mean) / self.input.shape[-1]) + \
                         d_mean / self.input.shape[-1]
        
        return input_gradient

    def get_trainable_params(self) -> List[Tuple[np.ndarray, Any]]:
        """Returns list of (parameter_array, gradient) tuples for LayerNorm. Training needs to happen for gradients to exist."""
        return [(self.gamma, 'gamma_gradient'), (self.beta, 'beta_gradient')]


class SelfAttention(Layer):
    """A small, single-head self-attention layer implemented in NumPy.
    Input shape: (batch, seq_len, dim). Returns same shape.
    not a full transformer.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Initialize projection matrices
        self.Wq = (np.random.randn(dim, dim).astype(FLOATX) * (np.sqrt(2.0 / dim)))
        self.Wk = (np.random.randn(dim, dim).astype(FLOATX) * (np.sqrt(2.0 / dim)))
        self.Wv = (np.random.randn(dim, dim).astype(FLOATX) * (np.sqrt(2.0 / dim)))
        self.Wo = (np.random.randn(dim, dim).astype(FLOATX) * (np.sqrt(2.0 / dim)))

        # Placeholder gradients (not fully implemented for training-heavy use)
        self.Wq_grad = None
        self.Wk_grad = None
        self.Wv_grad = None
        self.Wo_grad = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        # Expect input shape (batch, seq_len, dim)
        x = input_data.astype(FLOATX)
        self.input = x
        b, s, d = x.shape
        # Linear projections
        Q = np.dot(x, self.Wq)  # (b, s, d)
        K = np.dot(x, self.Wk)
        V = np.dot(x, self.Wv)

        # Scaled dot-product attention
        scale = FLOATX(np.sqrt(d))
        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / scale  # (b, s, s)

        # Stable softmax along last axis
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)

        context = np.matmul(attn, V)  # (b, s, d)

        out = np.dot(context, self.Wo)
        self.output = out.astype(FLOATX)
        self._attn = attn
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Lightweight backward: approximate by passing gradient through to inputs.
        # Full backprop for attention is costly; for now return gradient propagated by Wo^T.
        out_grad = output_gradient.astype(FLOATX)
        # Propagate through Wo
        input_grad = np.dot(out_grad, self.Wo.T)
        return input_grad

    def get_trainable_params(self) -> List[Tuple[np.ndarray, Any]]:
        return [(self.Wq, 'Wq_grad', self), (self.Wk, 'Wk_grad', self), (self.Wv, 'Wv_grad', self), (self.Wo, 'Wo_grad', self)]

class Sequential(Layer):
    """A sequential neural network container that chains layers together."""
    
    def __init__(self, *layers, learning_rate: float = 1e-4, **kwargs):
        super().__init__()
        self.layers = list(layers)
        if len(self.layers) == 1 and isinstance(self.layers[0], (list, tuple)):
            self.layers = list(self.layers[0])
        self.learning_rate = FLOATX(learning_rate)
        self.is_training = False

    def forward(self, input_data):
        """Forward pass through all layers."""
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, output_gradient):
        """Backward pass through all layers in reverse order."""
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)
        return output_gradient

    def get_trainable_params(self):
        """Get all trainable parameters from layers."""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'get_trainable_params'):
                layer_params = layer.get_trainable_params()
                for param_tuple in layer_params:
                    # Add layer instance to the tuple
                    params.append(param_tuple + (layer,))
        return params

    def train_step(self, inputs, targets):
        """Perform a training step using integrated layer optimizations."""
        # Normalize inputs to prevent overflow - use float32 for stability
        inputs = inputs.astype(FLOATX)
        inputs = (inputs - np.mean(inputs)) / (np.std(inputs) + 1e-8)
        
        # Forward pass
        outputs = self.forward(inputs)
        
        # Ensure outputs are float32 for loss computation
        outputs = outputs.astype(FLOATX)
        targets = targets.astype(FLOATX)
        
        # Compute loss (MSE) with NaN checking
        loss = np.mean((outputs - targets) ** 2)
        
        # Check for invalid loss
        if np.isnan(loss) or np.isinf(loss):
            print(f"Invalid loss detected: {loss}, outputs range: [{np.min(outputs):.6f}, {np.max(outputs):.6f}], targets range: [{np.min(targets):.6f}, {np.max(targets):.6f}]")
            return loss  # Return the invalid loss to be caught by caller
        
        # Backward pass (layers handle their own gradient modulation)
        gradients = (outputs - targets).astype(FLOATX)
        self.backward(gradients)
        
        # Simple SGD update for all layers with gradient clipping
        for param_array, grad_name, layer_instance in self.get_trainable_params():
            grad_array = getattr(layer_instance, grad_name)
            if grad_array is not None:
                # Clip gradients to prevent explosion
                grad_array = np.clip(grad_array, -10.0, 10.0)
                # Update parameters
                param_array -= self.learning_rate * grad_array.astype(param_array.dtype)
                # Reset gradient
                setattr(layer_instance, grad_name, None)
        
        return loss

    def get_performance_metrics(self):
        """Get current performance metrics from the monitor."""
        return self.performance_monitor.get_metrics()


class JustinJOptimizer:
    """
    An optimizer designed to develop self-awareness and agency through:
    1. Audio-motor coupling (connecting voice output to self-perception)
    2. Intentional control development
    3. Goal-oriented behavior emergence
    4. Multi-timescale learning
    """
    def __init__(self,
                 networks: List,
                 base_lr: float = 1e-4,
                 vocal_feedback_weight: float = 0.3,
                 agency_growth_rate: float = 0.01,
                 control_precision: float = 0.1,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.01):

        # Initialize base JustinJ optimizer
        self.justinj = AdamW(networks,
                           learning_rate=base_lr,
                           beta1=beta1,
                           beta2=beta2,
                           epsilon=epsilon,
                           weight_decay=weight_decay)

        self.networks = networks
        self.base_lr = FLOATX(base_lr)
        self.vocal_feedback_weight = FLOATX(vocal_feedback_weight)
        self.agency_growth_rate = FLOATX(agency_growth_rate)
        self.control_precision = FLOATX(control_precision)

        # Agency potential tracking
        self.p = {}  # Agency potential (how much a parameter influences agency)

        # Agency development tracking
        self.voice_control_confidence = 0.0  # Grows as voice control improves
        self.intention_alignment = 0.0       # Grows as actions match intentions
        self.feedback_recognition = 0.0      # Grows as system recognizes own outputs

        # Multi-timescale learning rates
        self.fast_lr = self.base_lr * 2.0   # For immediate feedback
        self.slow_lr = self.base_lr * 0.5   # For pattern consolidation

        # Temporal context windows
        self.short_term_memory = []  # Recent input-output pairs
        self.pattern_memory = []     # Successful control patterns

        self.t = 0  # Timestep counter
        
        # Gradient clipping to prevent exploding gradients
        self.max_grad_norm = 1.0  # Maximum gradient norm
        self.clip_gradients = True  # Enable gradient clipping

        logger = logging.getLogger(__name__)
        logger.info("JustinJOptimizer initialized with focus on developing self-awareness and control")

    def update_metrics(self,
                      vocal_output: np.ndarray,
                      audio_feedback: np.ndarray,
                      intended_output: Optional[np.ndarray] = None,
                      reward_signal: Optional[float] = None):
        """
        Update internal metrics based on system feedback
        """
        # Calculate voice feedback alignment
        if vocal_output is not None and audio_feedback is not None:
            feedback_match = self._calculate_feedback_match(vocal_output, audio_feedback)
            self.feedback_recognition = (self.feedback_recognition * 0.95 +
                                      feedback_match * 0.05)

        # Update intention alignment if provided
        if intended_output is not None and vocal_output is not None:
            control_accuracy = self._calculate_control_accuracy(intended_output, vocal_output)
            self.intention_alignment = (self.intention_alignment * 0.9 +
                                     control_accuracy * 0.1)

        # Update voice control confidence
        self.voice_control_confidence = (self.feedback_recognition * 0.6 +
                                       self.intention_alignment * 0.4)

        # Store context for pattern learning
        self._update_temporal_context(vocal_output, audio_feedback, intended_output, reward_signal)

    def _calculate_feedback_match(self, vocal_output: np.ndarray, audio_feedback: np.ndarray) -> float:
        """Calculate how well audio feedback matches expected output"""
        # Normalize both signals
        vocal_norm = vocal_output / (np.max(np.abs(vocal_output)) + 1e-8)
        feedback_norm = audio_feedback / (np.max(np.abs(audio_feedback)) + 1e-8)

        # Calculate correlation
        correlation = np.corrcoef(vocal_norm.flatten(), feedback_norm.flatten())[0,1]
        return max(0, correlation)  # Only positive correlation matters for recognition

    def _calculate_control_accuracy(self, intended: np.ndarray, actual: np.ndarray) -> float:
        """Calculate how well actual output matches intentions"""
        error = np.mean(np.square(intended - actual))
        return np.exp(-error / self.control_precision)

    def _update_temporal_context(self, vocal_output, audio_feedback, intended_output, reward):
        """Update temporal context windows for pattern learning"""
        context = {
            'time': time.time(),
            'vocal_output': vocal_output,
            'audio_feedback': audio_feedback,
            'intended_output': intended_output,
            'reward': reward
        }

        # Update short-term memory
        self.short_term_memory.append(context)
        if len(self.short_term_memory) > 100:  # Keep last 100 timesteps
            self.short_term_memory.pop(0)

        # Update pattern memory if this was a successful interaction
        if reward is not None and reward > 0.7:  # High reward threshold
            self.pattern_memory.append(context)
            if len(self.pattern_memory) > 1000:  # Keep last 1000 successful patterns
                self.pattern_memory.pop(0)

    def step(self):
        """Perform an optimization step combining JustinJ algorithm with agency development"""
        self.t += 1

        # Calculate agency-based learning rate modulation
        agency_factor = np.sqrt(self.voice_control_confidence + 0.1)

        # Temporarily modify JustinJ's learning rate based on agency development
        original_lr = self.justinj.lr
        self.justinj.lr = original_lr * agency_factor

        for net_idx, net in enumerate(self.networks):
            for param_array, grad_name, layer_instance in net.get_trainable_params():
                grad_array = getattr(layer_instance, grad_name)
                if grad_array is None:
                    continue

                param_key = (id(net), id(layer_instance), grad_name)

                # Initialize agency potential if needed
                if param_key not in self.p:
                    self.p[param_key] = np.ones_like(param_array, dtype=FLOATX)

                # Update agency potential
                if self.voice_control_confidence > self.p[param_key].mean():
                    # Parameter helped improve agency - strengthen it
                    self.p[param_key] *= (1.0 + self.agency_growth_rate)
                else:
                    # Parameter might be interfering - weaken it slightly
                    self.p[param_key] *= (1.0 - self.agency_growth_rate * 0.1)

                # Modify gradients based on agency potential and temporal context
                if len(self.pattern_memory) > 0:
                    # More emphasis on established patterns
                    grad_array *= np.sqrt(self.p[param_key]) * 1.2
                else:
                    # More emphasis on exploration
                    grad_array *= np.sqrt(self.p[param_key]) * 0.8

                # Update the gradient in the layer instance for JustinJ
                setattr(layer_instance, grad_name, grad_array)

        # Apply gradient clipping before JustinJ step
        if self.clip_gradients:
            self._clip_gradients()

        # Let JustinJ perform its optimization step
        self.justinj.step()

        # Restore original learning rate
        self.justinj.lr = original_lr

        # Log progress periodically
        if self.t % 100 == 0:
            logger = logging.getLogger(__name__)
            logger.info(f"Agency Metrics - Voice Control: {self.voice_control_confidence:.3f}, "
                       f"Intention Alignment: {self.intention_alignment:.3f}, "
                       f"Feedback Recognition: {self.feedback_recognition:.3f}")

    def get_agency_metrics(self) -> Dict[str, float]:
        """Return current agency development metrics"""
        return {
            'voice_control_confidence': float(self.voice_control_confidence),
            'intention_alignment': float(self.intention_alignment),
            'feedback_recognition': float(self.feedback_recognition),
            'pattern_memory_size': len(self.pattern_memory)
        }

    def _clip_gradients(self):
        """Clip gradients to prevent exploding gradients"""
        total_norm = 0.0
        
        # Calculate total gradient norm across all parameters
        for net in self.networks:
            for param_array, grad_name, layer_instance in net.get_trainable_params():
                grad_array = getattr(layer_instance, grad_name)
                if grad_array is not None:
                    param_norm = np.linalg.norm(grad_array)
                    total_norm += param_norm ** 2
        
        total_norm = np.sqrt(total_norm)
        
        # Clip gradients if norm exceeds threshold
        if total_norm > self.max_grad_norm:
            clip_coef = self.max_grad_norm / (total_norm + 1e-6)
            
            # Apply clipping to all gradients
            for net in self.networks:
                for param_array, grad_name, layer_instance in net.get_trainable_params():
                    grad_array = getattr(layer_instance, grad_name)
                    if grad_array is not None:
                        clipped_grad = grad_array * clip_coef
                        setattr(layer_instance, grad_name, clipped_grad)

# ==============================================================================
# SECTION 4: LOSS FUNCTIONS
# ==============================================================================

def mse_loss_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Derivative of Mean Squared Error loss."""
    return 2 * (y_pred - y_true) / y_true.size

# ==============================================================================
# TIER 2: EMBEDDED WEIGHT SAVE/LOAD (Module Self-Contained)
# ==============================================================================

# >>> EMBEDDED_NN_WEIGHTS
EMBEDDED_NN_WEIGHTS = {
    'initialized': False,
    'layer_count': 0,
    'layers': []
}
# <<< EMBEDDED_NN_WEIGHTS

def save_embedded_nn_weights(sequential_net: Sequential) -> bool:
    """
    Save Sequential network weights to embedded EMBEDDED_NN_WEIGHTS dict.
    This persists the trained state within nn.py itself.
    """
    global EMBEDDED_NN_WEIGHTS
    try:
        weights_data = {
            'initialized': True,
            'layer_count': len(sequential_net.layers),
            'layers': []
        }
        
        for i, layer in enumerate(sequential_net.layers):
            layer_state = layer.get_state() if hasattr(layer, 'get_state') else {}
            
            # For Linear layers, save weights and biases
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                layer_state['weights'] = layer.weights.tolist() if hasattr(layer.weights, 'tolist') else layer.weights
                layer_state['biases'] = layer.biases.tolist() if hasattr(layer.biases, 'tolist') else layer.biases
            
            weights_data['layers'].append(layer_state)
        
        EMBEDDED_NN_WEIGHTS = weights_data
        return True
    except Exception as e:
        print(f"Error saving embedded NN weights: {e}")
        return False

def load_embedded_nn_weights(sequential_net: Sequential) -> bool:
    """
    Load Sequential network weights from embedded EMBEDDED_NN_WEIGHTS dict.
    Restores the last trained state of the network.
    """
    global EMBEDDED_NN_WEIGHTS
    try:
        if not EMBEDDED_NN_WEIGHTS.get('initialized', False):
            return False
        
        if EMBEDDED_NN_WEIGHTS['layer_count'] != len(sequential_net.layers):
            print(f"Layer count mismatch: saved {EMBEDDED_NN_WEIGHTS['layer_count']}, current {len(sequential_net.layers)}")
            return False
        
        for i, layer in enumerate(sequential_net.layers):
            if i >= len(EMBEDDED_NN_WEIGHTS['layers']):
                break
            
            layer_state = EMBEDDED_NN_WEIGHTS['layers'][i]
            
            # Restore weights and biases for Linear layers
            if hasattr(layer, 'weights') and 'weights' in layer_state:
                layer.weights = np.array(layer_state['weights'], dtype=FLOATX)
            if hasattr(layer, 'biases') and 'biases' in layer_state:
                layer.biases = np.array(layer_state['biases'], dtype=FLOATX)
            
            # Restore state if set_state exists
            if hasattr(layer, 'set_state') and 'state' in layer_state:
                layer.set_state(layer_state['state'])
        
        return True
    except Exception as e:
        print(f"Error loading embedded NN weights: {e}")
        return False
