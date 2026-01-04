# som.py

import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional, Tuple 
from config import FLOATX, SOM_INPUT_DIM, SOM_MAP_ROWS, SOM_MAP_COLS, SOM_BMU_COORD_DIM, SOM_LEARNING_RATE, SOM_SIGMA, SOM_PLASTICITY_DECAY, SOM_PLASTICITY_MOD, SOM_FATIGUE_COST, SOM_FATIGUE_DECAY, SOM_ACTIVATION_THRESHOLD, SOM_INHIBITION_RADIUS_FACTOR, SOM_GROWTH_THRESHOLD, SOM_GROWTH_COOLDOWN, SOM_INHIBITION_STRENGTH

# Single consolidated state file for all modules

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SelfOrganizingMap:
    """
    A Complex Adaptive System (CAS) SOM for foundational cognitive processing.
    Designed with parameters geared towards productive fragility and organic learning.
    Supports dynamic growth through staged size increases.
    """
    
    # Disable dynamic growth: single fixed stage -> no growth will occur
    GROWTH_STAGES = []  # Will be set dynamically from config
    
    @staticmethod
    def _get_available_compute():
        """Calculate available compute resources for dynamic scaling"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            return (100 - cpu_percent) / 100, (100 - memory_percent) / 100
        except ImportError:
            return 0.5, 0.5  # Conservative defaults - training needed for optimal compute scaling. See config.py SOM_TRAINING
    
    def __init__(self,
                 map_size: tuple = None,  # Now optional, will use config if None
                 input_dim: int = None,  # Now optional, will use config if None
                 learning_rate: float = SOM_LEARNING_RATE,
                 sigma: float = SOM_SIGMA,
                 plasticity_decay: float = SOM_PLASTICITY_DECAY,
                 plasticity_mod: float = SOM_PLASTICITY_MOD,
                 fatigue_cost: float = SOM_FATIGUE_COST,
                 fatigue_decay: float = SOM_FATIGUE_DECAY,
                 activation_threshold: float = SOM_ACTIVATION_THRESHOLD,
                 inhibition_radius_factor: float = SOM_INHIBITION_RADIUS_FACTOR,
                 growth_threshold: float = SOM_GROWTH_THRESHOLD,  # High utilization triggers growth
                 growth_cooldown: int = SOM_GROWTH_COOLDOWN,    # Minimum cycles between growth
                 inhibition_strength: float = SOM_INHIBITION_STRENGTH):
        """
        Initializes the CAS SOM with parameters for emergent complexity and failure learning.

        Args:
            map_size (tuple): (height, width) of the neuron grid. If None, uses config.SOM_MAP_ROWS/SOM_MAP_COLS.
            input_dim (int): Dimensionality of the input vectors. If None, uses config.SOM_INPUT_DIM.
            learning_rate (float): Initial global learning rate.
            sigma (float): Initial radius of the excitatory neighborhood.
            plasticity_decay (float): Rate at which plasticity decays towards baseline.
            plasticity_mod (float): Amount plasticity increases after activation.
            fatigue_cost (float): Energy lost by BMU per learning event.
            fatigue_decay (float): Rate of energy regeneration for neurons.
            activation_threshold (float): Max distance for a neuron to be a BMU.
            inhibition_radius_factor (float): Multiplier for inhibitory radius vs. excitatory sigma.
            inhibition_strength (float): Magnitude of lateral inhibition.
        """
        # Load dimensions from config if not provided
        from config import SOM_INPUT_DIM, SOM_MAP_ROWS, SOM_MAP_COLS, SOM_GROWTH_STAGES
        if input_dim is None:
            input_dim = SOM_INPUT_DIM
        if map_size is None:
            map_size = (SOM_MAP_ROWS, SOM_MAP_COLS)
        
        # Set GROWTH_STAGES dynamically from config
        self.GROWTH_STAGES = SOM_GROWTH_STAGES if SOM_GROWTH_STAGES else [map_size]
        
        self.map_height = np.int32(map_size[0])
        self.map_width = np.int32(map_size[1])
        self.map_size = (self.map_height, self.map_width)
        self.input_dim = np.int32(input_dim)
        
        # --- Core Parameters ---
        self.initial_lr = FLOATX(learning_rate)
        self.initial_sigma = FLOATX(sigma)
        self.activation_threshold = FLOATX(activation_threshold)
        self.inhibition_radius_factor = FLOATX(inhibition_radius_factor)
        self.inhibition_strength = FLOATX(inhibition_strength)
        
        # --- Plasticity Configuration ---
        self.plasticity_baseline = FLOATX(1.0)
        self.plasticity_decay = FLOATX(plasticity_decay)
        self.plasticity_mod = FLOATX(plasticity_mod)

        # --- Fatigue Configuration ---
        self.fatigue_baseline = FLOATX(1.0)
        self.fatigue_cost = FLOATX(fatigue_cost)
        self.fatigue_decay = FLOATX(fatigue_decay)

        # --- Internal State Matrices ---
        # Initialize weights with small random values within [0, 1] to match normalized inputs
        self.weights = np.random.uniform(0, 1, (self.map_height, self.map_width, self.input_dim)).astype(FLOATX)
        # Plasticity initialized to baseline
        self.plasticity = np.full(self.map_size, self.plasticity_baseline, dtype=FLOATX)
        # Fatigue initialized to baseline (fully charged)
        self.fatigue = np.full(self.map_size, self.fatigue_baseline, dtype=FLOATX)
        
        # --- Pre-computed Static Data ---
        # Neuron coordinates for distance calculations
        self.neuron_coords = np.indices(self.map_size).transpose(1, 2, 0).astype(FLOATX)
        self.last_update_time = time.perf_counter() # For fatigue decay calculation

        # --- Growth Parameters ---
        self.growth_threshold = growth_threshold
        self.growth_cooldown = growth_cooldown
        self.current_growth_stage = 0  # Index into GROWTH_STAGES
        self.cycles_since_growth = 0
        self.utilization_history = []
        
        # --- Failure State Variables ---
        self.critical_threshold_exceeded = False # Flag for external systems
        self.failure_log = [] # Stores data upon critical state detection

        # Initialize logging data structures
        self.growth_log = []
        self.performance_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'learning_times': [],
            'quantization_errors': [],
            'growth_events': [],
            'plasticity_stats': [],
            'fatigue_stats': [],
            'utilization_stats': []
        }
        
        # Log initialization
        init_log = {
            'timestamp': time.time(),
            'event': 'startup',
            'map_size': map_size,
            'input_dim': input_dim,
            'params': {
                'learning_rate': learning_rate,
                'sigma': sigma,
                'plasticity_decay': plasticity_decay,
                'fatigue_decay': fatigue_decay,
                'activation_threshold': activation_threshold
            }
        }
        self.growth_log.append(init_log)
        logger.info(f"CAS SOM started: Map={map_size}, InputDim={input_dim}")
    # Usage tracking structures will be initialized lazily on first learning call
    # (process_and_learn sets bmu_heatmap, neuron_age, last_activation as needed).

    def _log_performance_metrics(self):
        """Log current performance metrics"""
        cpu_avail, mem_avail = self._get_available_compute()
        current_metrics = {
            'timestamp': time.time(),
            'cpu_usage': 1.0 - cpu_avail,
            'memory_usage': 1.0 - mem_avail,
            'map_size': self.map_size,
            'quantization_error': np.mean(self.quantization_errors[-50:]) if hasattr(self, 'quantization_errors') and self.quantization_errors else None,
            'plasticity_mean': float(np.mean(self.plasticity)),
            'plasticity_std': float(np.std(self.plasticity)),
            'fatigue_mean': float(np.mean(self.fatigue)),
            'fatigue_std': float(np.std(self.fatigue)),
            'utilization': float(np.sum(self.bmu_heatmap > 0) / (self.map_height * self.map_width))
        }
        
        # Update metrics history
        for key, value in current_metrics.items():
            if key in self.performance_metrics:
                self.performance_metrics[key].append(value)
                
        # Keep only last 1000 entries for each metric
        for key in self.performance_metrics[key]:
            if len(self.performance_metrics[key]) > 1000:
                self.performance_metrics[key] = self.performance_metrics[key][-1000:]
        
        return current_metrics

    def _check_growth_conditions(self) -> bool:
        """Check if the SOM should grow based on multiple factors and available resources"""
        if self.current_growth_stage >= len(self.GROWTH_STAGES) - 1:
            return False  # Already at maximum size
            
        if self.cycles_since_growth < self.growth_cooldown:
            return False  # Still in cooldown period
            
        # 1. Calculate recent utilization
        recent_utilization = np.mean(self.utilization_history[-100:]) if self.utilization_history else 0
        
        # 2. Check BMU distribution (clustering)
        bmu_distribution = self.bmu_heatmap / (np.sum(self.bmu_heatmap) + 1e-6)
        entropy = -np.sum(bmu_distribution * np.log2(bmu_distribution + 1e-6))
        max_entropy = np.log2(self.map_height * self.map_width)
        distribution_uniformity = entropy / max_entropy
        
        # 3. Check average quantization error trend
        recent_errors = self.quantization_errors[-50:] if hasattr(self, 'quantization_errors') else []
        error_trend = np.mean(np.diff(recent_errors)) if len(recent_errors) > 1 else 0
        
        # 4. Check neighborhood plasticity
        avg_plasticity = np.mean(self.plasticity)
        plasticity_variance = np.var(self.plasticity)
        
        # 5. Check available compute resources
        cpu_avail, mem_avail = self._get_available_compute()
        resource_score = min(cpu_avail, mem_avail)
        
        # Adaptive growth threshold based on resources
        adaptive_threshold = self.growth_threshold * (1.0 + (1.0 - resource_score))
        
        # Growth conditions with resource awareness:
        needs_growth = (
            resource_score > 0.3 and (  # Only grow if sufficient resources
                (recent_utilization > adaptive_threshold) or              # High utilization (adaptive)
                (distribution_uniformity < 0.7 and avg_plasticity < 0.3 and resource_score > 0.5) or  # Poor distribution
                (error_trend > 0 and recent_utilization > adaptive_threshold * 0.8) or  # Rising errors
                (plasticity_variance < 0.1 and recent_utilization > adaptive_threshold * 0.9)  # Uniform plasticity
            )
        )
        
        return needs_growth

    def _grow_map(self):
        """Grow the SOM to the next size stage while preserving learned patterns"""
        if not self._check_growth_conditions():
            return
            
        old_size = self.map_size
        new_size = self.GROWTH_STAGES[self.current_growth_stage + 1]
        
        # Store old weights and states
        old_weights = self.weights.copy()
        old_plasticity = self.plasticity.copy()
        old_fatigue = self.fatigue.copy()
        
        # Initialize new arrays
        self.map_height, self.map_width = new_size
        self.map_size = new_size
        self.weights = np.random.uniform(-1, 1, (self.map_height, self.map_width, self.input_dim)).astype(FLOATX)
        self.plasticity = np.full(self.map_size, self.plasticity_baseline, dtype=FLOATX)
        self.fatigue = np.full(self.map_size, self.fatigue_baseline, dtype=FLOATX)
        
        # Interpolate old values into new arrays
        y_indices = np.linspace(0, old_size[0]-1, new_size[0])
        x_indices = np.linspace(0, old_size[1]-1, new_size[1])
        for i, y in enumerate(y_indices):
            for j, x in enumerate(x_indices):
                y1, y2 = int(np.floor(y)), int(np.ceil(y))
                x1, x2 = int(np.floor(x)), int(np.ceil(x))
                
                if y1 == y2: y2 = min(y1 + 1, old_size[0] - 1)
                if x1 == x2: x2 = min(x1 + 1, old_size[1] - 1)
                
                # Bilinear interpolation weights
                wy2 = y - y1
                wy1 = 1 - wy2
                wx2 = x - x1
                wx1 = 1 - wx2
                
                # Interpolate weights
                self.weights[i, j] = (wy1 * (wx1 * old_weights[y1, x1] + wx2 * old_weights[y1, x2]) +
                                    wy2 * (wx1 * old_weights[y2, x1] + wx2 * old_weights[y2, x2]))
                                    
                # Interpolate plasticity and fatigue
                self.plasticity[i, j] = (wy1 * (wx1 * old_plasticity[y1, x1] + wx2 * old_plasticity[y1, x2]) +
                                       wy2 * (wx1 * old_plasticity[y2, x1] + wx2 * old_plasticity[y2, x2]))
                self.fatigue[i, j] = (wy1 * (wx1 * old_fatigue[y1, x1] + wx2 * old_fatigue[y1, x2]) +
                                    wy2 * (wx1 * old_fatigue[y2, x1] + wx2 * old_fatigue[y2, x2]))
        
        # Update coordinates for distance calculations
        self.neuron_coords = np.indices(self.map_size).transpose(1, 2, 0).astype(FLOATX)
        
        # Update growth state
        self.current_growth_stage += 1
        self.cycles_since_growth = 0
        
        # Initialize tracking for new size
        self.bmu_heatmap = np.zeros(self.map_size, dtype=np.float32)
        self.neuron_age = np.zeros(self.map_size, dtype=np.float32)
        self.last_activation = np.full(self.map_size, -np.inf, dtype=np.float32)
        
        # Log growth event with detailed metrics
        growth_event = {
            'timestamp': time.time(),
            'event': 'growth',
            'old_size': old_size,
            'new_size': new_size,
            'metrics_before': self._log_performance_metrics(),
            'growth_stage': self.current_growth_stage,
            'resource_state': {
                'cpu_available': self._get_available_compute()[0],
                'memory_available': self._get_available_compute()[1]
            }
        }
        self.growth_log.append(growth_event)
        self.performance_metrics['growth_events'].append(growth_event)
        
        logger.info(f"SOM grew from {old_size} to {new_size}")

    def _prune_unused_neurons(self, current_time: float):
        """Identify and reset unused or ineffective neurons"""
        # Don't prune if we haven't had any activations yet (last_activation starts as -inf)
        if np.max(self.last_activation) < 0:  # All -inf means no activations yet
            return
        
        # Calculate time since last activation for each neuron
        activation_age = current_time - self.last_activation
        
        # Identify neurons that haven't been active recently
        inactive_mask = activation_age > (self.growth_cooldown * 0.5)
        low_utility_mask = self.bmu_heatmap < np.mean(self.bmu_heatmap) * 0.1
        
        neurons_to_reset = inactive_mask & low_utility_mask
        
        if np.any(neurons_to_reset):
            # Reset these neurons with slight variations of neighboring active neurons
            active_indices = np.where(~neurons_to_reset)
            reset_indices = np.where(neurons_to_reset)
            
            for reset_y, reset_x in zip(*reset_indices):
                # Find nearest active neuron
                distances = np.sqrt(
                    (active_indices[0] - reset_y)**2 + 
                    (active_indices[1] - reset_x)**2
                )
                nearest_idx = np.argmin(distances)
                nearest_y, nearest_x = active_indices[0][nearest_idx], active_indices[1][nearest_idx]
                
                # Reset to nearest active neuron - variation should come from training
                # See config.py SOM_TRAINING section for proper reset strategies
                self.weights[reset_y, reset_x] = self.weights[nearest_y, nearest_x].copy()
                self.plasticity[reset_y, reset_x] = self.plasticity_baseline
                self.fatigue[reset_y, reset_x] = self.fatigue_baseline
                self.neuron_age[reset_y, reset_x] = 0
                
            logger.info(f"Reset {np.sum(neurons_to_reset)} unused neurons")

    def _find_bmu(self, input_vector: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Finds the Best Matching Unit (BMU), considering neuronal fatigue. 
        Returns None if no neuron is a sufficiently good match (below threshold).
        """
        # Fatigue integration: fatigued neurons (low fatigue value) are less responsive.
        # We model this by increasing their effective distance, making them less likely BMUs.
        # Add a small epsilon to prevent division by zero if fatigue is exactly 0.
        fatigue_penalty = 1.0 / (self.fatigue + 1e-6) 
        
        # Calculate squared Euclidean distance between input and all weights
        distance_sq = np.sum(np.square(self.weights - input_vector.reshape((1, 1, self.input_dim))), axis=2)
        
        # Apply fatigue penalty to distances
        fatigue_adjusted_distance = distance_sq * fatigue_penalty

        # Find the index of the minimum adjusted distance
        min_dist_idx = np.argmin(fatigue_adjusted_distance)
        bmu_coords = np.unravel_index(min_dist_idx, self.map_size)
        # Convert to builtin ints for consistency
        bmu_coords = (int(bmu_coords[0]), int(bmu_coords[1]))

        # Activation Threshold: If the best match is still too far, no effective BMU.
        # Use the unadjusted distance for the threshold check.
        if np.sqrt(distance_sq[bmu_coords]) > self.activation_threshold:
            return None

        return bmu_coords

    def _update_internal_states(self, bmu_coords: tuple, excitatory_influence: np.ndarray):
        """
        Updates dynamic internal state matrices (fatigue, plasticity) and handles
        temporal decay for lingering SOM vectors. Called after a learning event.
        """
        time_delta = time.perf_counter() - self.last_update_time
        
        # --- Update Fatigue ---
        # All neurons regenerate energy over time
        self.fatigue += self.fatigue_decay * time_delta
        # The winning neuron (if any) expends energy
        if bmu_coords is not None:
            self.fatigue[bmu_coords] -= self.fatigue_cost
        # Clip fatigue to prevent negative values and excessive energy
        self.fatigue = np.clip(self.fatigue, 0.01, self.fatigue_baseline)

        # --- Update Plasticity ---
        # Plasticity decays towards baseline for all neurons
        self.plasticity *= (1.0 - self.plasticity_decay * time_delta) # Scale decay by time
        # Neurons influenced by learning become more plastic
        if bmu_coords is not None:
            plasticity_increase = excitatory_influence * self.plasticity_mod
            self.plasticity += plasticity_increase
        # Clip plasticity to prevent extreme values
        self.plasticity = np.clip(self.plasticity, 0.1, 5.0)

        # --- Temporal Decay on Lingering SOM Vectors ---
        # The decay of plasticity and increase in fatigue on less active neurons
        # serve as implicit temporal decay. No additional explicit decay of weights themselves
        # is added here, as it's typically handled by the learning rule and neighborhood function
        # making weights of inactive neurons less influenced over time.
        
        self.last_update_time = time.perf_counter()

    def _check_and_log_failure(self, input_vector: np.ndarray, bmu_coords: tuple):
        """Checks for failure conditions and logs data if a critical threshold is met."""
        
        # Condition 1: BMU fatigue is critically low
        if bmu_coords is not None and self.fatigue[bmu_coords] < 0.01:  # Lower threshold; avoid frequent triggers
            # Log the condition but do not set the global critical suppression flag.
            logger.warning(f"Low BMU fatigue detected for {bmu_coords}: {self.fatigue[bmu_coords]:.4f}")

        # Condition 2: Very poor match despite finding a BMU (input is very far from learned patterns)
        if bmu_coords is not None:
            bmu_distance = np.sqrt(np.sum(np.square(self.weights[bmu_coords] - input_vector)**2))
            if bmu_distance > self.activation_threshold * 1.5: # 1.5x the normal activation threshold
                # Log the condition for diagnostics but do not force a critical suppression.
                logger.warning(f"Poor BMU match at {bmu_coords} (dist={bmu_distance:.2f}); continuing operation.")
        
        # Condition 3: No BMU found at all (input is completely novel or unmappable)
        if bmu_coords is None:
            logger.warning("No Best Matching Unit found for input. Input may be novel or unmappable; continuing.")

    def visualize_growth(self) -> dict:
        """Generate visualization data for the SOM's state"""
        current_time = time.perf_counter()
        
        viz_data = {
            'map_size': self.map_size,
            'utilization': {
                'current': np.sum(self.bmu_heatmap > 0) / (self.map_height * self.map_width),
                'history': self.utilization_history[-100:] if self.utilization_history else []
            },
            'neuron_health': {
                'activity_heatmap': self.bmu_heatmap.copy(),
                'age_map': self.neuron_age.copy(),
                'plasticity_map': self.plasticity.copy(),
                'fatigue_map': self.fatigue.copy()
            },
            'growth_metrics': {
                'stage': self.current_growth_stage,
                'next_size': self.GROWTH_STAGES[min(self.current_growth_stage + 1, len(self.GROWTH_STAGES) - 1)],
                'cycles_since_growth': self.cycles_since_growth,
                'growth_readiness': self._calculate_growth_readiness()
            }
        }
        return viz_data
        
    def _calculate_growth_readiness(self) -> float:
        """Calculate a normalized score (0-1) indicating readiness for growth"""
        if self.current_growth_stage >= len(self.GROWTH_STAGES) - 1:
            return 0.0
            
        recent_utilization = np.mean(self.utilization_history[-100:]) if self.utilization_history else 0
        time_factor = min(self.cycles_since_growth / self.growth_cooldown, 1.0)
        plasticity_health = np.mean(self.plasticity) / self.plasticity_baseline
        
        readiness = (
            0.4 * recent_utilization +
            0.3 * time_factor +
            0.3 * plasticity_health
        )
        return float(readiness)

    def process_and_learn(self, data_batch: list, learning_bias: float = 1.0) -> np.ndarray:
        """
        Processes a batch of data, performs learning, and returns activation map.
        Handles 'productive failures' by logging state and flagging the critical condition.
        Learning for an input is skipped if a critical state is detected for that input.
        """
        if not data_batch:
            return np.zeros(self.map_size, dtype=np.float32)

        # Lazy-init usage tracking structures if missing (backward compatibility)
        if not hasattr(self, 'bmu_heatmap'):
            self.bmu_heatmap = np.zeros(self.map_size, dtype=np.float32)
        if not hasattr(self, 'neuron_age'):
            self.neuron_age = np.zeros(self.map_size, dtype=np.float32)
        if not hasattr(self, 'last_activation'):
            self.last_activation = np.full(self.map_size, -np.inf, dtype=np.float32)

        iterations = len(data_batch)
        activation_map = np.zeros(self.map_size, dtype=np.float32)
        current_time = time.perf_counter()
        
        # The critical_threshold_exceeded flag should be reset by an *external* recovery mechanism
        # after it processes the failure. It reflects the current internal state of SOM.

        for i, input_vector in enumerate(data_batch):
            input_vector = input_vector.astype(np.float32)
            
            # --- Find BMU and Check for Failure ---
            bmu_coords = self._find_bmu(input_vector)
            self._check_and_log_failure(input_vector, bmu_coords)

            # If a critical state is detected or no BMU found, skip learning but update time-based states
            if bmu_coords is None or self.critical_threshold_exceeded:
                self._update_internal_states(bmu_coords, np.zeros(self.map_size))
                continue

            # --- Valid BMU found, proceed with learning ---
            # Calculate learning parameters that decay over the batch
            sigma_decayed = self.initial_sigma * (1.0 - i / iterations)  # Linear decay
            
            # Calculate distances and influences
            distance_sq = np.sum(np.square(self.neuron_coords - np.array(bmu_coords)), axis=2)
            
            # Calculate complex neighborhood functions
            excitatory_influence = np.exp(-distance_sq / (2 * np.square(sigma_decayed)))
            inhibitory_sigma = sigma_decayed * self.inhibition_radius_factor
            inhibitory_influence = self.inhibition_strength * np.exp(-distance_sq / (2 * np.square(inhibitory_sigma)))
            
            # Compute plasticity-modulated learning influence
            plasticity_influence = self.plasticity * (excitatory_influence - inhibitory_influence)
            
            # Calculate adaptive learning rate based on multiple factors
            local_lr = self.initial_lr * learning_bias * (
                (1.0 - i/iterations) *  # Time-based decay
                np.clip(self.fatigue[bmu_coords], 0.1, 1.0) *  # Fatigue modulation
                (1.0 + self.plasticity[bmu_coords] * 0.2)  # Plasticity boost
            )
            
            # Update weights with complex learning rule
            weight_delta = (
                local_lr * 
                plasticity_influence[..., np.newaxis] * 
                (input_vector - self.weights)
            )
            self.weights += weight_delta.astype(np.float32)
            
            # Update internal states and record activation
            self._update_internal_states(bmu_coords, excitatory_influence)
            activation_map[bmu_coords] += 1

        # Update neuron usage statistics
        self.bmu_heatmap += activation_map
        active_mask = activation_map > 0
        self.last_activation[active_mask] = current_time
        self.neuron_age += (current_time - self.last_update_time)
        
        # Calculate utilization for this batch
        utilization = np.sum(activation_map > 0) / (self.map_height * self.map_width)
        self.utilization_history.append(utilization)
        if len(self.utilization_history) > 1000:  # Keep last 1000 batches
            self.utilization_history.pop(0)
            
        # Track quantization errors
        if not hasattr(self, 'quantization_errors'):
            self.quantization_errors = []
        if len(data_batch) > 0:
            errors = [np.min(np.linalg.norm(self.weights - x.reshape(1, 1, -1), axis=2))
                     for x in data_batch]
            self.quantization_errors.append(np.mean(errors))
            if len(self.quantization_errors) > 1000:
                self.quantization_errors.pop(0)
        
        # Prune unused neurons periodically
        if self.cycles_since_growth % 100 == 0:
            self._prune_unused_neurons(current_time)
            
        # Increment cycles and check for growth
        self.cycles_since_growth += 1
        if self._check_growth_conditions():
            self._grow_map()
            
        self.last_update_time = current_time
        return activation_map
            

    def get_activation_map(self, data_batch: list) -> np.ndarray:
        """Calculates the activation pattern of the SOM for a given batch without learning."""
        activation_map = np.zeros(self.map_size, dtype=np.float32)
        if not data_batch: return activation_map
        
        # Reset the critical_threshold_exceeded flag for this inference pass
        self.critical_threshold_exceeded = False 

        for vector in data_batch:
            vector = vector.astype(np.float32)
            bmu_coords = self._find_bmu(vector)
            # For pure activation map generation, we don't log failures or check critical state
            # but we still need to update internal states (like fatigue for future BMU finding)
            self._update_internal_states(bmu_coords, np.zeros(self.map_size)) # No learning, so zero influence
            if bmu_coords is not None:
                activation_map[bmu_coords] += 1
        return activation_map
        
                # if 'weights' in state and np.array(state['weights']).shape == self.weights.shape:
                #     self.weights = np.array(state['weights'], dtype=np.float32)
                   
                   
            
    def get_plasticity_map(self) -> np.ndarray:
        """Returns the current plasticity values for every neuron."""
        return self.plasticity.copy()
        
    def get_fatigue_map(self) -> np.ndarray:
        """Returns the current fatigue (energy) level for every neuron."""
        return self.fatigue.copy()
        
    def get_map_entropy(self) -> float:
        """Calculates the entropy of the map's weights, a measure of representation diversity."""
        # Flatten weights and calculate histogram for distribution analysis
        flat_weights = self.weights.flatten()
        # Use density=True to get probabilities, then remove zeros
        hist, _ = np.histogram(flat_weights, bins=256, density=True)
        hist = hist[hist > 0] # Remove zero probabilities to avoid log(0)
        
        # Shannon entropy calculation: -sum(p * log2(p))
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
        
    def get_bmu_map(self, data_batch: list) -> np.ndarray:
        """Generates a map indicating how many times each neuron was a BMU for the data batch."""
        bmu_map = np.zeros(self.map_size, dtype=np.int32)
        if not data_batch: return bmu_map
        
        for vector in data_batch:
            vector = vector.astype(np.float32)
            bmu_coords = self._find_bmu(vector)
            if bmu_coords is not None:
                bmu_map[bmu_coords] += 1
        return bmu_map

    def get_quantization_error(self, data_batch: list) -> float:
        """Calculates the average distance between input vectors and their BMU weights."""
        total_error, n_samples = 0.0, 0
        if not data_batch: return 0.0
        
        for vector in data_batch:
            vector = vector.astype(np.float32)
            bmu_coords = self._find_bmu(vector)
            if bmu_coords is not None:
                bmu_weight = self.weights[bmu_coords]
                # Calculate Euclidean distance
                error = np.linalg.norm(vector - bmu_weight)
                total_error += error
                n_samples += 1
        
        # Return average error, or 0 if no samples mapped
        return (total_error / n_samples) if n_samples > 0 else 0.0

    # ---------------- Training Status / Introspection ----------------
    def get_training_status(self) -> dict:
        """Heuristic assessment of whether the SOM has been meaningfully trained.

        A SOM is considered 'trained' (heuristically) if:
          - We have a BMU heatmap with sufficient total hits ( > 5 * number of neurons )
          - Utilization (fraction of neurons that have been BMU at least once) exceeds 20%
          - Recent quantization error metric exists (was computed during learning cycles)
        These thresholds are adjustable and meant only for runtime sanity checks / logging.
        """
        total_neurons = int(self.map_height * self.map_width)
        total_bmu_hits = float(np.sum(self.bmu_heatmap)) if hasattr(self, 'bmu_heatmap') else 0.0
        utilization = float(
            np.sum(self.bmu_heatmap > 0) / total_neurons
        ) if hasattr(self, 'bmu_heatmap') and total_neurons > 0 else 0.0
        recent_qe = float(np.mean(self.quantization_errors[-10:])) if hasattr(self, 'quantization_errors') and self.quantization_errors else None
        trained = (
            total_bmu_hits > (5 * total_neurons) and
            utilization > 0.20 and
            recent_qe is not None
        )
        return {
            'trained': trained,
            'total_bmu_hits': total_bmu_hits,
            'utilization_fraction': utilization,
            'recent_quant_error': recent_qe,
            'map_size': self.map_size,
            'input_dim': int(self.input_dim),
            'neuron_count': total_neurons
        }

    # --- Backward Compatibility Adapter ---
    def process_input(self, input_vector: np.ndarray) -> np.ndarray:
        """Legacy single-vector processing interface expected by main.py.

        Accepts a 1D feature vector, performs a learning step (single item batch)
        using the existing process_and_learn path to keep internal statistics
        consistent, then returns a flattened activation map (map_height*map_width,).

        If the input is invalid it returns a zero activation vector.
        """
        try:
            if input_vector is None:
                logger.error("SOM.process_input received None as input_vector. Returning zero activation vector.")
                return np.zeros(self.map_height * self.map_width, dtype=np.float32)
            # Ensure 1D float32
            input_vec = np.array(input_vector, dtype=np.float32).reshape(-1)
            # Basic dimension guard – if mismatch, attempt truncate/pad
            if input_vec.shape[0] != self.input_dim:
                logger.warning(f"SOM.process_input received input_vector with mismatched dimensions. Expected {self.input_dim}, got {input_vec.shape[0]}. Attempting to adjust.")
                if input_vec.shape[0] > self.input_dim:
                    input_vec = input_vec[:self.input_dim]
                else:
                    input_vec = np.pad(input_vec, (0, self.input_dim - input_vec.shape[0]))
            activation_map = self.process_and_learn([input_vec])
            return activation_map.astype(np.float32).flatten()
        except Exception as e:
            logger.exception("Exception in SOM.process_input: {e}")
            return np.zeros(self.map_height * self.map_width, dtype=np.float32)

# >>> EMBEDDED_SOM_WEIGHTS
EMBEDDED_SOM_WEIGHTS = {
    'initialized': False,
    'map_size': (SOM_MAP_ROWS, SOM_MAP_COLS),
    'weights': None,
    'plasticity_matrix': None,
    'fatigue_map': None
}
# <<< EMBEDDED_SOM_WEIGHTS

def save_embedded_som_weights(som_instance) -> bool:
    """
    Save SOM weights to embedded EMBEDDED_SOM_WEIGHTS dict.
    Persists the trained SOM state within som.py itself.
    """
    global EMBEDDED_SOM_WEIGHTS
    try:
        EMBEDDED_SOM_WEIGHTS = {
            'initialized': True,
            'map_size': (som_instance.map_height, som_instance.map_width),
            'weights': som_instance.weights.tolist() if hasattr(som_instance.weights, 'tolist') else som_instance.weights,
            'plasticity_matrix': som_instance.plasticity_matrix.tolist() if hasattr(som_instance.plasticity_matrix, 'tolist') else som_instance.plasticity_matrix,
            'fatigue_map': som_instance.fatigue_map.tolist() if hasattr(som_instance.fatigue_map, 'tolist') else som_instance.fatigue_map
        }
        return True
    except Exception as e:
        logger.error(f"Error saving embedded SOM weights: {e}")
        return False

def load_embedded_som_weights(som_instance) -> bool:
    """
    Load SOM weights from embedded EMBEDDED_SOM_WEIGHTS dict.
    Restores the last trained state of the SOM.
    """
    global EMBEDDED_SOM_WEIGHTS
    try:
        if not EMBEDDED_SOM_WEIGHTS.get('initialized', False):
            return False
        
        if EMBEDDED_SOM_WEIGHTS['weights'] is not None:
            som_instance.weights = np.array(EMBEDDED_SOM_WEIGHTS['weights'], dtype=FLOATX)
        if EMBEDDED_SOM_WEIGHTS['plasticity_matrix'] is not None:
            som_instance.plasticity_matrix = np.array(EMBEDDED_SOM_WEIGHTS['plasticity_matrix'], dtype=FLOATX)
        if EMBEDDED_SOM_WEIGHTS['fatigue_map'] is not None:
            som_instance.fatigue_map = np.array(EMBEDDED_SOM_WEIGHTS['fatigue_map'], dtype=FLOATX)
        
        return True
    except Exception as e:
        logger.error(f"Error loading embedded SOM weights: {e}")
        return False

# >>> EMBEDDED_SOM_STATE
EMBEDDED_SOM_STATE = {}
# <<< EMBEDDED_SOM_STATE
