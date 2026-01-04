# JustinJ_Optimizer.py
# Standalone copy of JustinJOptimizer with internal audio feedback loop scaffolding.
# Retained original version still inside nn.py; this file allows portability & future divergence.

import numpy as np
import logging
import time
from collections import deque
from typing import List, Dict, Optional, Tuple, Any
import threading

# Optional torch import for torch-based parameter updates
try:
    import torch
    from torch.nn.utils import clip_grad_norm_
    _HAS_TORCH = True
except Exception:
    torch = None
    clip_grad_norm_ = None
    _HAS_TORCH = False

# Optional DirectML for AMD/Intel GPUs on Windows
try:
    import torch_directml
    _HAS_DML = True
except Exception:
    torch_directml = None
    _HAS_DML = False


try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    psutil = None
    _HAS_PSUTIL = False


class PerformanceMonitor:
    """
    Performance monitoring utility to track memory usage, CPU load, and operation timings.
    """

    def __init__(self):
        self.start_time = time.time()
        self.operation_times: Dict[str, List[float]] = {}
        self.memory_usage: List[float] = []
        self.cpu_usage: List[float] = []
        self._lock = threading.Lock()
        self.monitoring = False
        self._thread = None
        if _HAS_PSUTIL:
            self.monitoring = True
            self._thread = threading.Thread(target=self._monitor_system, daemon=True)
            self._thread.start()

    def _monitor_system(self):
        while self.monitoring:
            try:
                proc = psutil.Process()
                mem = proc.memory_info().rss / 1024.0 / 1024.0
                cpu = proc.cpu_percent(interval=None)
                with self._lock:
                    self.memory_usage.append(mem)
                    self.cpu_usage.append(cpu)
                time.sleep(1.0)
            except Exception:
                break

    def track_operation(self, operation_name: str, category: Optional[str] = None, duration: Optional[float] = None):
        """Record an operation duration (if provided) or reserve the operation name."""
        with self._lock:
            if operation_name not in self.operation_times:
                self.operation_times[operation_name] = []
            if duration is not None:
                try:
                    self.operation_times[operation_name].append(float(duration))
                except Exception:
                    pass

    def time_operation(self, operation_name: str):
        """Decorator to time functions and record durations."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                res = func(*args, **kwargs)
                t1 = time.perf_counter()
                self.track_operation(operation_name, None, t1 - t0)
                return res
            return wrapper
        return decorator

    def get_performance_summary(self) -> Dict[str, Any]:
        """Return a compact summary (safe when psutil is absent)."""
        with self._lock:
            avg_mem = float(np.mean(self.memory_usage)) if self.memory_usage else 0.0
            peak_mem = float(np.max(self.memory_usage)) if self.memory_usage else 0.0
            avg_cpu = float(np.mean(self.cpu_usage)) if self.cpu_usage else 0.0
            op_stats = {}
            for op, times in self.operation_times.items():
                if times:
                    op_stats[op] = {
                        'count': len(times),
                        'total_time': float(sum(times)),
                        'average_time': float(np.mean(times)),
                        'max_time': float(np.max(times)),
                        'min_time': float(np.min(times))
                    }
                else:
                    op_stats[op] = {'count': 0}

            return {
                'uptime_seconds': time.time() - self.start_time,
                'average_memory_mb': avg_mem,
                'peak_memory_mb': peak_mem,
                'average_cpu_percent': avg_cpu,
                'operation_stats': op_stats,
                'psutil_available': _HAS_PSUTIL
            }

    def stop_monitoring(self):
        self.monitoring = False
        if self._thread is not None and self._thread.is_alive():
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

class JustinJOptimizer:
    """Agency-focused optimizer with integrated vocal/audio internal feedback loop.

    Core Additions vs baseline in nn.py:
    - Optional internal echo loop if real mic input missing (still prefers real mic)
    - Audio alignment metrics (correlation, spectral centroid alignment)
    - Latent intention reconstruction (predict intended control vector from audio echo)
    - Multi-timescale adaptive LR modulation
    - Gradient hygiene (clipping + nan/inf scrubbing)
    - Pattern memory with decay + replay sampling hooks
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
                 weight_decay: float = 0.01,
                 enable_internal_echo: bool = True,
                 spectral_weight: float = 0.2,
                 replay_capacity: int = 512,
                 fast_mode: bool = False,
                 audio_out: Optional[Any] = None,
                 audio_in: Optional[Any] = None,
                 enable_performance_monitoring: bool = True):
        # Core setup (internal JustinJ optimizer state; fully decoupled from external optimizer)
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor() if enable_performance_monitoring else None
        
        self.networks = networks or []
        self.base_lr = float(base_lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)  # decoupled JustinJ style
        self.vocal_feedback_weight = float(vocal_feedback_weight)
        self.agency_growth_rate = float(agency_growth_rate)
        self.control_precision = float(control_precision)
        self.enable_internal_echo = enable_internal_echo
        self.spectral_weight = float(spectral_weight)
        self.replay_capacity = replay_capacity
        self.fast_mode = bool(fast_mode)
        self.audio_out = audio_out
        self.audio_in = audio_in

        # Agency metrics
        self.voice_control_confidence = 0.0
        self.intention_alignment = 0.0
        self.feedback_recognition = 0.0
        self.spectral_alignment = 0.0

        # Adaptive learning buffers
        self.p = {}
        self.t = 0

        # Memories
        self.short_term_memory = []
        self.pattern_memory = []
        self.replay_buffer = []

        # Gradient safety
        self.max_grad_norm = 1.0
        self.clip_gradients = True

        # Cached last intended output for echo fallback
        self._last_intended_output = None
        self._last_audio_feedback = None

        # Extended adaptive scheduling
        self.warmup_steps = 500
        self.cooldown_patience = 800
        self.min_lr = self.base_lr * 0.1
        self.max_lr = self.base_lr * 5.0
        self.last_improvement_step = 0
        self.improvement_threshold = 0.002
        self.metric_history = deque(maxlen=400)

        # Temporal coherence tracking
        self._ema_vocal = None
        self.temporal_coherence = 0.0

        # Additional spectral stats
        self.spectral_flatness = 0.0
        self.spectral_bandwidth = 0.0

        # Prioritized replay
        self.priority_replay = []
        self.priority_alpha = 0.7
        self.priority_epsilon = 1e-4

        # Plateau detection buffers
        self._rolling_vc = deque(maxlen=100)
        self._rolling_alignment = deque(maxlen=100)

        # Gradient freeze map
        self.frozen_params = {}
        self.freeze_duration = 300
        self.freeze_threshold = 0.0005
        self.max_freeze_fraction = 0.15

        # Moment & variance state per parameter key
        self.m = {}
        self.v = {}
        self._bias_correction_cache = {}

        logger.info("JustinJOptimizer (standalone) initialized with performance monitoring (enriched + integrated JustinJ algorithm).")

    # --------------------------- Audio / Feedback Utilities ---------------------------
    def _normalize_audio(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        m = np.max(np.abs(x)) + 1e-8
        return x / m

    def _spectral_centroid(self, signal: np.ndarray, sr: int = 44100) -> float:
        signal = signal.astype(np.float32)
        if signal.size == 0:
            return 0.0
        # FFT
        spec = np.fft.rfft(signal)
        mag = np.abs(spec)
        freqs = np.fft.rfftfreq(signal.size, 1.0 / sr)
        denom = np.sum(mag) + 1e-8
        return float(np.sum(freqs * mag) / denom)

    def _spectral_flatness_bandwidth(self, signal: np.ndarray, sr: int = 44100) -> Tuple[float, float]:
        signal = signal.astype(np.float32)
        if signal.size == 0:
            return 0.0, 0.0
        spec = np.fft.rfft(signal)
        mag = np.abs(spec) + 1e-12
        geo_mean = np.exp(np.mean(np.log(mag)))
        arith_mean = np.mean(mag)
        flatness = float(geo_mean / (arith_mean + 1e-12))
        freqs = np.fft.rfftfreq(signal.size, 1.0 / sr)
        centroid = self._spectral_centroid(signal, sr)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / (np.sum(mag) + 1e-12)))
        return flatness, bandwidth


# --------------------------- Vectorized Operations ---------------------------
class VectorizedOperations:
    """
    Collection of vectorized operations to replace slow loops.
    """

    @staticmethod
    def batch_matrix_multiply(a: np.ndarray, b: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        if a.shape[0] <= batch_size:
            return np.dot(a, b)

        result = np.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)
        for i in range(0, a.shape[0], batch_size):
            end_idx = min(i + batch_size, a.shape[0])
            result[i:end_idx] = np.dot(a[i:end_idx], b)

        return result

    @staticmethod
    def vectorized_activation(activations: np.ndarray, activation_type: str = 'tanh') -> np.ndarray:
        if activation_type == 'tanh':
            return np.tanh(activations)
        elif activation_type == 'relu':
            return np.maximum(0, activations)
        elif activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-activations))
        elif activation_type == 'leaky_relu':
            return np.where(activations > 0, activations, 0.01 * activations)
        else:
            return activations

    @staticmethod
    def optimized_convolution(input_data: np.ndarray, kernels: np.ndarray,
                            stride: int = 1, padding: int = 0) -> np.ndarray:
        batch_size, input_height, input_width, input_channels = input_data.shape
        num_kernels, kernel_height, kernel_width, _ = kernels.shape

        output_height = (input_height + 2 * padding - kernel_height) // stride + 1
        output_width = (input_width + 2 * padding - kernel_width) // stride + 1

        if padding > 0:
            input_data = np.pad(input_data, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')

        output = np.zeros((batch_size, output_height, output_width, num_kernels), dtype=input_data.dtype)

        for i in range(output_height):
            for j in range(output_width):
                h_start, h_end = i * stride, i * stride + kernel_height
                w_start, w_end = j * stride, j * stride + kernel_width

                receptive_field = input_data[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.sum(receptive_field[..., np.newaxis] * kernels, axis=(1, 2, 3))

        return output

    @staticmethod
    def memory_efficient_batch_norm(input_data: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        mean = np.mean(input_data, axis=0, keepdims=True)
        variance = np.var(input_data, axis=0, keepdims=True)
        normalized = input_data - mean
        normalized /= np.sqrt(variance + epsilon)
        return normalized


# --------------------------- Memory Pool ---------------------------
class MemoryPool:
    def __init__(self, max_pool_size: int = 1000):
        self.pool: Dict[str, List[np.ndarray]] = {}
        self.max_pool_size = max_pool_size
        self.pool_stats = {'hits': 0, 'misses': 0, 'allocations': 0}

    def get_array(self, shape: Tuple, dtype: np.dtype = np.float32, key: Optional[str] = None) -> np.ndarray:
        if key is None:
            key = f"{shape}_{dtype}"

        if key in self.pool and self.pool[key]:
            self.pool_stats['hits'] += 1
            array = self.pool[key].pop()
            if array.shape != shape:
                array = np.resize(array, shape)
            return array
        else:
            self.pool_stats['misses'] += 1
            self.pool_stats['allocations'] += 1
            return np.zeros(shape, dtype=dtype)

    def return_array(self, array: np.ndarray, key: Optional[str] = None):
        if key is None:
            key = f"{array.shape}_{array.dtype}"

        if key not in self.pool:
            self.pool[key] = []

        if len(self.pool[key]) < self.max_pool_size:
            array.fill(0)
            self.pool[key].append(array)

    def get_stats(self) -> Dict[str, Any]:
        total_arrays = sum(len(arrays) for arrays in self.pool.values())
        return {
            **self.pool_stats,
            'total_arrays_in_pool': total_arrays,
            'pool_keys': list(self.pool.keys())
        }

    def clear_pool(self):
        self.pool.clear()
        import gc
        gc.collect()


# --------------------------- Optimized Neural Network ---------------------------
class OptimizedNeuralNetwork:
    def __init__(self, memory_pool: Optional[MemoryPool] = None):
        self.memory_pool = memory_pool or MemoryPool()
        self.performance_monitor = PerformanceMonitor()
        self.vector_ops = VectorizedOperations()
        self.temp_buffers = {'activation': None, 'gradient': None, 'weight_update': None}

    def optimized_forward(self, layers: List, input_data: np.ndarray) -> np.ndarray:
        start_time = time.perf_counter()
        x = input_data
        for layer in layers:
            if hasattr(layer, 'forward'):
                if self.temp_buffers['activation'] is None or self.temp_buffers['activation'].shape != x.shape:
                    if self.temp_buffers['activation'] is not None:
                        self.memory_pool.return_array(self.temp_buffers['activation'], 'activation')
                    self.temp_buffers['activation'] = self.memory_pool.get_array(x.shape, x.dtype, 'activation')
                x = layer.forward(x)
        end_time = time.perf_counter()
        duration = end_time - start_time
        if not hasattr(self, 'forward_times'):
            self.forward_times = []
        self.forward_times.append(duration)
        return x

    def optimized_backward(self, layers: List, output_gradient: np.ndarray) -> np.ndarray:
        start_time = time.perf_counter()
        grad = output_gradient
        for layer in reversed(layers):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)
        end_time = time.perf_counter()
        duration = end_time - start_time
        if not hasattr(self, 'backward_times'):
            self.backward_times = []
        self.backward_times.append(duration)
        return grad

    def get_performance_report(self) -> Dict[str, Any]:
        return {
            'memory_pool_stats': self.memory_pool.get_stats(),
            'performance_monitor': self.performance_monitor.get_performance_summary(),
            'optimization_efficiency': self._calculate_efficiency()
        }

    def _calculate_efficiency(self) -> float:
        pool_stats = self.memory_pool.get_stats()
        if pool_stats['hits'] + pool_stats['misses'] > 0:
            hit_rate = pool_stats['hits'] / (pool_stats['hits'] + pool_stats['misses'])
            return hit_rate
        return 0.0

    def cleanup(self):
        self.memory_pool.clear_pool()
        self.performance_monitor.stop_monitoring()
        import gc
        gc.collect()


# Lazy singletons (avoid heavy cost at import time)
_GLOBAL_MEMORY_POOL: Optional[MemoryPool] = None
_GLOBAL_VECTOR_OPS: Optional[VectorizedOperations] = None
_GLOBAL_OPTIMIZED_NN: Optional[OptimizedNeuralNetwork] = None

def get_memory_pool() -> MemoryPool:
    global _GLOBAL_MEMORY_POOL
    if _GLOBAL_MEMORY_POOL is None:
        _GLOBAL_MEMORY_POOL = MemoryPool()
    return _GLOBAL_MEMORY_POOL

def get_vector_ops() -> VectorizedOperations:
    global _GLOBAL_VECTOR_OPS
    if _GLOBAL_VECTOR_OPS is None:
        _GLOBAL_VECTOR_OPS = VectorizedOperations()
    return _GLOBAL_VECTOR_OPS

def get_optimized_nn() -> OptimizedNeuralNetwork:
    global _GLOBAL_OPTIMIZED_NN
    if _GLOBAL_OPTIMIZED_NN is None:
        _GLOBAL_OPTIMIZED_NN = OptimizedNeuralNetwork(get_memory_pool())
    return _GLOBAL_OPTIMIZED_NN

    def _calculate_feedback_match(self, vocal_output: np.ndarray, audio_feedback: np.ndarray) -> float:
        v = self._normalize_audio(vocal_output).flatten()
        a = self._normalize_audio(audio_feedback).flatten()
        if v.size != a.size:
            n = min(v.size, a.size)
            v = v[:n]; a = a[:n]
        if v.size == 0:
            return 0.0
        corr = np.corrcoef(v, a)[0,1]
        if not np.isfinite(corr):
            corr = 0.0
        return max(0.0, float(corr))

    def _calculate_control_accuracy(self, intended: np.ndarray, actual: np.ndarray) -> float:
        if intended is None or actual is None or intended.size == 0 or actual.size == 0:
            return 0.0
        n = min(intended.size, actual.size)
        err = np.mean((intended[:n] - actual[:n])**2)
        return float(np.exp(-err / max(self.control_precision,1e-6)))

    def _spectral_alignment_metric(self, vocal_output: np.ndarray, audio_feedback: np.ndarray) -> float:
        c1 = self._spectral_centroid(vocal_output)
        c2 = self._spectral_centroid(audio_feedback)
        diff = abs(c1 - c2)
        return float(np.exp(-diff / 800.0))  # heuristic scale

    # --------------------------- Public Metric Update ---------------------------
    def update_metrics(self,
                       vocal_output: Optional[np.ndarray],
                       audio_feedback: Optional[np.ndarray],
                       intended_output: Optional[np.ndarray] = None,
                       reward_signal: Optional[float] = None,
                       sample_rate: int = 44100):
        # Performance monitoring
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            self.performance_monitor.track_operation("update_metrics", "feedback_processing")
            
        # Fallback echo if no mic capture yet
        if (audio_feedback is None or audio_feedback.size == 0) and self.enable_internal_echo:
            audio_feedback = vocal_output.copy() if vocal_output is not None else self._last_audio_feedback
        if vocal_output is None and self._last_intended_output is not None:
            vocal_output = self._last_intended_output  # crude fallback

        if vocal_output is None or audio_feedback is None:
            return

        # Core correlations
        fb_match = self._calculate_feedback_match(vocal_output, audio_feedback)
        self.feedback_recognition = self.feedback_recognition * 0.95 + fb_match * 0.05

        if intended_output is not None:
            ctrl_acc = self._calculate_control_accuracy(intended_output, vocal_output)
            self.intention_alignment = self.intention_alignment * 0.9 + ctrl_acc * 0.1
        else:
            ctrl_acc = 0.0

        if not self.fast_mode:
            spec_align = self._spectral_alignment_metric(vocal_output, audio_feedback)
            self.spectral_alignment = self.spectral_alignment * 0.9 + spec_align * 0.1

            # Additional spectral stats
            flatness, bandwidth = self._spectral_flatness_bandwidth(vocal_output, sample_rate)
            self.spectral_flatness = 0.95 * self.spectral_flatness + 0.05 * flatness
            self.spectral_bandwidth = 0.95 * self.spectral_bandwidth + 0.05 * bandwidth
        else:
            spec_align = 0.0
            flatness = 0.0
            bandwidth = 0.0

        # Temporal coherence: similarity of current output to EMA
        if self._ema_vocal is None:
            self._ema_vocal = vocal_output.astype(np.float32)
        else:
            # update ema
            self._ema_vocal = 0.9 * self._ema_vocal + 0.1 * vocal_output.astype(np.float32)
            n = min(self._ema_vocal.size, vocal_output.size)
            if n > 0:
                num = np.dot(self._ema_vocal[:n], vocal_output[:n])
                den = (np.linalg.norm(self._ema_vocal[:n]) * np.linalg.norm(vocal_output[:n]) + 1e-8)
                coh = num / den
                if np.isfinite(coh):
                    self.temporal_coherence = 0.95 * self.temporal_coherence + 0.05 * coh

        self.voice_control_confidence = (
            0.5 * self.feedback_recognition + 0.3 * self.intention_alignment + 0.2 * self.spectral_alignment
        )

        context = {
            'time': time.time(),
            'vocal_output': vocal_output.copy(),
            'audio_feedback': audio_feedback.copy(),
            'intended_output': None if intended_output is None else intended_output.copy(),
            'reward': reward_signal,
            'fb_match': fb_match,
            'ctrl_acc': ctrl_acc,
            'spec_align': spec_align,
            'spec_flat': flatness,
            'spec_band': bandwidth,
            'temp_coh': self.temporal_coherence
        }
        self.short_term_memory.append(context)
        if len(self.short_term_memory) > 128:
            self.short_term_memory.pop(0)

        if reward_signal is not None and reward_signal > 0.7:
            self.pattern_memory.append(context)
            if len(self.pattern_memory) > 512:
                self.pattern_memory.pop(0)

        # Replay store (simple chronological)
        self.replay_buffer.append(context)
        if len(self.replay_buffer) > self.replay_capacity:
            self.replay_buffer.pop(0)

        # Priority replay (priority derived from combined metrics + reward)
        priority = (
            0.4 * fb_match + 0.3 * self.intention_alignment + 0.2 * spec_align + 0.1 * self.temporal_coherence
        )
        if reward_signal is not None:
            priority += 0.3 * reward_signal
        if not self.fast_mode:
            self.priority_replay.append((float(priority), context))
            if len(self.priority_replay) > self.replay_capacity:
                self.priority_replay.pop(0)

        # Track improvement history for schedule decisions
        self.metric_history.append(self.voice_control_confidence)
        self._rolling_vc.append(self.voice_control_confidence)
        self._rolling_alignment.append(self.intention_alignment)

        # Detect improvements
        if len(self.metric_history) > 5:
            recent = list(self.metric_history)[-5:]
            if (max(recent) - min(recent)) > self.improvement_threshold:
                self.last_improvement_step = self.t

        self._last_intended_output = intended_output.copy() if intended_output is not None else self._last_intended_output
        self._last_audio_feedback = audio_feedback.copy()

    # --------------------------- Optimization Step ---------------------------
    def step(self):
        """Perform one optimization step (integrated JustinJ algorithm + adaptive agency scaling)."""
        # Performance monitoring
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            self.performance_monitor.track_operation("optimizer_step", "optimization")
        
        self.t += 1

        # ----- Learning rate scheduling -----
        agency_factor = np.sqrt(self.voice_control_confidence + 0.1)
        schedule_lr = self.base_lr
        if self.t < self.warmup_steps:  # warmup
            warm_frac = self.t / max(1, self.warmup_steps)
            schedule_lr = self.base_lr * (0.1 + 0.9 * warm_frac)
        elif (self.t - self.last_improvement_step) > self.cooldown_patience:  # cooldown plateau
            schedule_lr = max(self.min_lr, self.base_lr * 0.5)
        elif self.voice_control_confidence > 0.8:  # expansion when strong
            schedule_lr = min(self.max_lr, self.base_lr * (1.0 + (self.voice_control_confidence - 0.8) * 2.0))

        effective_lr = schedule_lr * agency_factor

        # ----- Parameter loop -----
        freeze_keys_considered = 0
        for net in self.networks:
            for param_array, grad_name, layer_instance in net.get_trainable_params():
                grad_array = getattr(layer_instance, grad_name)
                if grad_array is None:
                    continue
                key = (id(net), id(layer_instance), grad_name)

                # Initialize per-param state lazily
                if key not in self.p:
                    self.p[key] = np.ones_like(param_array, dtype=np.float32)
                if key not in self.m:
                    self.m[key] = np.zeros_like(param_array, dtype=np.float32)
                if key not in self.v:
                    self.v[key] = np.zeros_like(param_array, dtype=np.float32)

                # Skip if currently frozen
                if key in self.frozen_params and self.t < self.frozen_params[key]:
                    continue

                # ----- Agency potential update -----
                if self.voice_control_confidence > self.p[key].mean():
                    self.p[key] *= (1.0 + self.agency_growth_rate)
                else:
                    self.p[key] *= (1.0 - 0.1 * self.agency_growth_rate)

                # Exploration vs consolidation scaling
                if len(self.pattern_memory) > 0:
                    scale = np.sqrt(self.p[key]) * 1.15
                else:
                    scale = np.sqrt(self.p[key]) * 0.85
                if self.temporal_coherence > 0.6:  # damp updates if already stable
                    scale *= 0.9

                grad_array = grad_array * scale
                grad_array = np.nan_to_num(grad_array, nan=0.0, posinf=0.0, neginf=0.0)

                # Lightweight inactivity freeze heuristic
                if freeze_keys_considered < 500:
                    mean_abs = float(np.mean(np.abs(grad_array)))
                    if mean_abs < self.freeze_threshold and len(self.frozen_params) < int(self.max_freeze_fraction * 1000):
                        self.frozen_params[key] = self.t + self.freeze_duration
                    freeze_keys_considered += 1

                # ----- JustinJ moment updates -----
                m = self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad_array
                v = self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad_array * grad_array)
                # Bias corrections
                bc1 = 1 - self.beta1 ** self.t
                bc2 = 1 - self.beta2 ** self.t
                m_hat = m / (bc1 + 1e-12)
                v_hat = v / (bc2 + 1e-12)
                update = m_hat / (np.sqrt(v_hat) + self.epsilon)

                # Decoupled weight decay
                if self.weight_decay > 0.0:
                    param_array -= effective_lr * self.weight_decay * param_array

                # Apply update
                param_array -= effective_lr * update

        if self.clip_gradients:
            self._clip_gradients()

        if self.t % 200 == 0:
            logger.info(
                f"JustinJOpt t={self.t} lr_eff={effective_lr:.2e} VC={self.voice_control_confidence:.3f} "
                f"FB={self.feedback_recognition:.3f} IA={self.intention_alignment:.3f} SPEC={self.spectral_alignment:.3f} "
                f"Flat={self.spectral_flatness:.3f} Band={self.spectral_bandwidth:.1f} Coh={self.temporal_coherence:.3f} Replay={len(self.replay_buffer)}")

        # store last lr for diagnostics
        self._last_effective_lr = effective_lr

    # --------------------------- Prioritized Replay Utilities ---------------------------
    def _sample_priority_indices(self, batch_size: int) -> List[int]:
        if not self.priority_replay:
            return []
        priorities = np.array([p for p, _ in self.priority_replay], dtype=np.float32)
        probs = (priorities + self.priority_epsilon) ** self.priority_alpha
        probs /= probs.sum()
        count = min(batch_size, len(self.priority_replay))
        return list(np.random.choice(len(self.priority_replay), size=count, replace=False, p=probs))

    def optimize_with_replay(self, batch_size: int = 16):
        """Lightweight auxiliary adaptation using prioritized replay contexts.
        This does NOT backprop (no computational graph) but modulates agency potentials
        using stored high-priority alignment examples so future gradient scaling reflects them.
        """
        # Performance monitoring
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            self.performance_monitor.track_operation("replay_optimization", "memory_processing")
            
        if self.fast_mode:
            return
        idxs = self._sample_priority_indices(batch_size)
        if not idxs:
            return
        avg_fb = 0.0; avg_ctrl = 0.0; avg_spec = 0.0
        for i in idxs:
            priority, ctx = self.priority_replay[i]
            avg_fb += ctx['fb_match']
            avg_ctrl += ctx['ctrl_acc']
            avg_spec += ctx['spec_align']
        n = len(idxs)
        avg_fb /= n; avg_ctrl /= n; avg_spec /= n
        composite = 0.5 * avg_fb + 0.3 * avg_ctrl + 0.2 * avg_spec
        # Adjust global potentials subtly
        adjust = (composite - 0.5) * 0.01  # small influence
        for k in list(self.p.keys())[:200]:  # cap cost
            self.p[k] *= (1.0 + adjust)

    # --------------------------- Diagnostics ---------------------------
    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            **self.get_agency_metrics(),
            'spectral_flatness': float(self.spectral_flatness),
            'spectral_bandwidth': float(self.spectral_bandwidth),
            'temporal_coherence': float(self.temporal_coherence),
            'fast_mode': self.fast_mode,
            'last_effective_lr': getattr(self, '_last_effective_lr', None),
            'lr_base': float(self.base_lr),
            'frozen_params': len(self.frozen_params),
            'warmup_remaining': max(0, self.warmup_steps - self.t),
            'since_improvement': self.t - self.last_improvement_step
        }

    # --------------------------- Gradient Clipping ---------------------------
    def _clip_gradients(self):
        total_norm = 0.0
        for net in self.networks:
            for _, grad_name, layer in net.get_trainable_params():
                g = getattr(layer, grad_name)
                if g is not None:
                    total_norm += float(np.linalg.norm(g)**2)
        total_norm = np.sqrt(total_norm)
        if total_norm > self.max_grad_norm:
            coef = self.max_grad_norm / (total_norm + 1e-6)
            for net in self.networks:
                for _, grad_name, layer in net.get_trainable_params():
                    g = getattr(layer, grad_name)
                    if g is not None:
                        setattr(layer, grad_name, g * coef)

    # --------------------------- Replay Sampling Hook ---------------------------
    def sample_replay(self, batch_size: int = 8) -> List[Dict[str, Any]]:
        if not self.replay_buffer:
            return []
        idxs = np.random.choice(len(self.replay_buffer), size=min(batch_size, len(self.replay_buffer)), replace=False)
        return [self.replay_buffer[i] for i in idxs]

    def get_agency_metrics(self) -> Dict[str, float]:
        return {
            'voice_control_confidence': float(self.voice_control_confidence),
            'intention_alignment': float(self.intention_alignment),
            'feedback_recognition': float(self.feedback_recognition),
            'spectral_alignment': float(self.spectral_alignment),
            'pattern_memory_size': len(self.pattern_memory),
            'replay_size': len(self.replay_buffer),
            'temporal_coherence': float(self.temporal_coherence)
        }

    # --------------------------- Audio Playback and Feedback Integration --------------------------- 
    def play_vocal_and_update_feedback(self, vocal_output: np.ndarray, intended_output: Optional[np.ndarray] = None, reward_signal: Optional[float] = None):
        """Play vocal output through AudioOut and capture feedback from AudioIn for self-hearing loop."""
        if self.audio_out is not None:
            self.audio_out.play_audio(vocal_output)
            # Brief delay for playback and capture
            time.sleep(0.05)
        
        # Get audio feedback from AudioIn
        audio_feedback = None
        if self.audio_in is not None:
            
            try:
                # Use get_recent_samples or similar; adjust based on AudioIn API
                recent = self.audio_in._get_recent_samples(self.audio_in.frame_size)
                if recent and len(recent) > 0:
                    # Take the first channel or mix; convert to numpy array
                    audio_feedback = np.array(recent[0], dtype=np.float32)
            except Exception as e:
                logger.warning(f"Failed to get audio feedback from AudioIn: {e}")
        
        # Update metrics with the captured feedback
        self.update_metrics(vocal_output, audio_feedback, intended_output, reward_signal)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the integrated performance monitor."""
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            return self.performance_monitor.get_performance_summary()
        return {}

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status including performance metrics."""
        status = {
            'step': self.t,
            'voice_control_confidence': self.voice_control_confidence,
            'intention_alignment': self.intention_alignment,
            'feedback_recognition': self.feedback_recognition,
            'spectral_alignment': self.spectral_alignment,
            'temporal_coherence': self.temporal_coherence,
            'spectral_flatness': self.spectral_flatness,
            'spectral_bandwidth': self.spectral_bandwidth,
            'learning_rate': self.base_lr,
            'short_term_memory_size': len(self.short_term_memory),
            'pattern_memory_size': len(self.pattern_memory),
            'replay_buffer_size': len(self.replay_buffer),
            'frozen_params_count': len(self.frozen_params)
        }
        
        # Add performance metrics if monitor is available
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            perf_metrics = self.performance_monitor.get_performance_summary()
            status['performance_metrics'] = perf_metrics
            
        return status

# EOF


# --------------------------- Torch Adapter ---------------------------
class JustinJVoiceOptim:
    """Torch optimizer that reuses JustinJ voice/agency signals.

    This keeps the JustinJ audio/agency machinery intact while performing decoupled
    weight decay and biased moment updates directly on torch parameters. It does not
    depend on torch built-ins and can be swapped in place of standard optimizers for Athena.
    """

    def __init__(
        self,
        model,
        base_lr: float = 1e-4,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 500,
        cooldown_patience: int = 800,
        min_lr: Optional[float] = None,
        max_lr: Optional[float] = None,
        agency_source: Optional[JustinJOptimizer] = None,
    ):
        if not _HAS_TORCH:
            raise ImportError("JustinJVoiceOptim requires torch; install PyTorch first.")

        self.model = model
        self.base_lr = float(base_lr)
        self.weight_decay = float(weight_decay)
        self.beta1, self.beta2 = betas
        self.epsilon = float(epsilon)
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
        self.warmup_steps = int(warmup_steps)
        self.cooldown_patience = int(cooldown_patience)
        self.min_lr = float(min_lr) if min_lr is not None else self.base_lr * 0.1
        self.max_lr = float(max_lr) if max_lr is not None else self.base_lr * 5.0
        self.agency_source = agency_source  # optional JustinJOptimizer for voice metrics

        self.state: Dict[torch.nn.Parameter, Dict[str, torch.Tensor]] = {}
        self.t = 0
        self.last_improvement_step = 0

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def _agency_factor(self) -> float:
        if self.agency_source is None:
            return 1.0
        return float(np.sqrt(self.agency_source.voice_control_confidence + 0.1))

    def _scheduled_lr(self) -> float:
        schedule_lr = self.base_lr
        if self.t < self.warmup_steps:
            warm_frac = self.t / max(1, self.warmup_steps)
            schedule_lr = self.base_lr * (0.1 + 0.9 * warm_frac)
        elif (self.t - self.last_improvement_step) > self.cooldown_patience:
            schedule_lr = max(self.min_lr, self.base_lr * 0.5)
        elif self.agency_source is not None and self.agency_source.voice_control_confidence > 0.8:
            schedule_lr = min(
                self.max_lr,
                self.base_lr * (1.0 + (self.agency_source.voice_control_confidence - 0.8) * 2.0),
            )
        return schedule_lr * self._agency_factor()

    @torch.no_grad()
    def step(self):
        params = [p for p in self.model.parameters() if p.grad is not None]
        if not params:
            return

        self.t += 1

        # Optional gradient clipping
        if self.max_grad_norm is not None and self.max_grad_norm > 0 and clip_grad_norm_ is not None:
            clip_grad_norm_(params, self.max_grad_norm)

        lr_eff = self._scheduled_lr()

        for p in params:
            grad = p.grad
            if grad.is_sparse:
                grad = grad.coalesce().to_dense()

            state = self.state.setdefault(p, {})
            m = state.get('m')
            v = state.get('v')

            if m is None:
                m = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['m'] = m
            if v is None:
                v = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['v'] = v

            # Adam moments
            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            bc1 = 1.0 - self.beta1 ** self.t
            bc2 = 1.0 - self.beta2 ** self.t
            m_hat = m / bc1
            v_hat = v / bc2

            update = m_hat / (v_hat.sqrt().add(self.epsilon))

            # Decoupled weight decay
            if self.weight_decay > 0.0:
                p.data.add_(p.data, alpha=-lr_eff * self.weight_decay)

            p.data.add_(update, alpha=-lr_eff)

    def get_last_lr(self) -> float:
        return self._scheduled_lr()


def athena_train_step(
    model,
    optimizer: JustinJVoiceOptim,
    input_ids,
    decoder_input_ids,
    targets,
    input_mask=None,
    decoder_mask=None,
    images=None,
    device=None,
    require_dml: bool = True,
):
    """Single training step for Athena using JustinJVoiceAdamW.

    Expects that targets align with decoder logits shape from Athena.compute_loss.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for Athena training.")

    # Select DirectML GPU if available (preferred), else raise if require_dml is True
    if device is None:
        if require_dml:
            if not _HAS_DML:
                raise ImportError("torch-directml not available; install to use GPU.")
            device = torch_directml.device("DML")
        else:
            device = torch_directml.device("DML") if _HAS_DML else torch.device("cpu")

    model.to(device)

    def to_dev(x):
        return x.to(device) if x is not None else None

    input_ids = to_dev(input_ids)
    decoder_input_ids = to_dev(decoder_input_ids)
    targets = to_dev(targets)
    input_mask = to_dev(input_mask)
    decoder_mask = to_dev(decoder_mask)
    images = to_dev(images)

    model.train()
    optimizer.zero_grad()

    output = model(input_ids, decoder_input_ids, input_mask, decoder_mask, images)
    loss = model.compute_loss(output["decoder_logits"], targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())
