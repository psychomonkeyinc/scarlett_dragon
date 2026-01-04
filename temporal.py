# temporal.py
# Numpy-only minimal Temporal Fabric for integration with your nn.py and som.py
# Usage: put next to nn.py and som.py and import TemporalFabric

# temporal.py
# Numpy-only minimal Temporal Fabric for integration with your nn.py and som.py
# Usage: put next to nn.py and som.py and import TemporalFabric


#python


# temporal.py
# Numpy-only minimal Temporal Fabric for integration with your nn.py and som.py
# Usage: put next to nn.py and som.py and import TemporalFabric

import time
import numpy as np
from collections import deque, namedtuple
from typing import List, Dict, Any, Optional
from sklearn.decomposition import PCA

try:
    # optional, used only if available for better mid compression
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# Local data structure for mid segments
Segment = namedtuple("Segment", ["embedding", "emotion_sig", "novelty", "timestamp", "meta"])

class TemporalConfig:
    def __init__(self):
        # buffers & sizing
        self.N_short = 2048
        self.N_segment_steps = 256
        self.mid_emb_dim = 128

        # dilation / conv
        self.dilation_profile = [1, 2, 4, 8, 16]
        self.kernel_size = 3  # causal kernel length
        self.conv_channels = 64  # internal channel dim (affects projection size)

        # emotion traces (three taus in steps)
        self.emotion_taus = np.array([2.0, 20.0, 200.0], dtype=float)

        # neuromodulator init
        self.mod_init = {"dopamine": 1.0, "serotonin": 1.0, "norepi": 1.0, "oxytocin": 1.0}

        # energy/metabolism
        self.energy_max = 1000.0
        self.write_cost = 1.0
        self.compress_cost = 20.0
        self.energy_recover_rate = 1.0
        self.dream_interval_steps = 10000

        # retention scoring weights
        self.w_n = 1.0
        self.w_e = 1.0
        self.w_r = 1.0
        self.w_p = 1.0
        self.recency_tau = 1e4

        # soft caps
        self.mid_soft_cap = 5000

class DilatedConvLite:
    """
    Cheap dilated conv-like module implemented in NumPy.
    The 'conv' uses learned kernel tensors per dilation that operate across embedding dims.
    It produces an output vector per segment close operation.
    """
    def __init__(self, emb_dim: int, cfg: TemporalConfig):
        self.emb_dim = emb_dim
        self.cfg = cfg
        # For each dilation we hold a kernel of shape (kernel_size, emb_dim) and a bias vector
        self.kernels = {}
        self.biases = {}
        for d in cfg.dilation_profile:
            # small random init
            self.kernels[d] = (0.1 * np.random.randn(cfg.kernel_size, emb_dim)).astype(np.float16)
            self.biases[d] = np.zeros((emb_dim,), dtype=np.float16)

        # simple gating vector per dilation (learnable)
        self.gates = {d: (np.ones((emb_dim,), dtype=np.float16) * np.float16(0.5)) for d in cfg.dilation_profile}

    def forward_last(self, seq: List[np.ndarray]):
        """
        Compute last-step representation using each dilation over the most recent times.
        seq: list of embeddings (older...newer). Each embedding is 1D array emb_dim.
        Returns concatenated vector (len = emb_dim * num_dilations)
        """
        out_parts = []
        T = len(seq)
        for d in self.cfg.dilation_profile:
            k = self.kernels[d]  # (K, emb_dim)
            K = k.shape[0]
            # gather indices for kernel taps (causal)
            taps = []
            for ki in range(K):
                idx = T - 1 - ki * d
                if idx < 0:
                    taps.append(np.zeros(self.emb_dim, dtype=np.float16))
                else:
                    taps.append(seq[idx])
            taps = np.stack(taps, axis=0)  # (K, emb_dim)
            # conv-like: elementwise multiply then sum over kernel dimension
            conv_out = np.sum(taps * k, axis=0) + self.biases[d]
            # gate (simple sigmoid gating)
            g = 1.0 / (1.0 + np.exp(-self.gates[d]))
            conv_out = conv_out * g
            out_parts.append(conv_out)
        # concat parts
        return np.concatenate(out_parts, axis=0)  # length = emb_dim * num_dilations

class LinearCompressor:
    """
    Minimal linear compressor: project concatenated conv output down to mid_emb_dim.
    Optionally uses PCA if sklearn available and user passes use_pca=True.
    """
    def __init__(self, input_dim: int, out_dim: int, use_pca: bool = False):
        self.in_dim = input_dim
        self.out_dim = out_dim
        self.use_pca = use_pca and _HAS_SKLEARN
        if self.use_pca:
            self._pca = PCA(n_components=out_dim)
            self._pca_fitted = False
            self._pca_buffer = []
        else:
            # linear projection weights
            self.W = (0.1 * np.random.randn(input_dim, out_dim)).astype(np.float16)
            self.b = np.zeros((out_dim,), dtype=np.float16)

    def compress(self, x: np.ndarray):
        # x shape: (in_dim,)
        if self.use_pca:
            if not self._pca_fitted:
                # buffer until we have enough samples
                self._pca_buffer.append(x)
                if len(self._pca_buffer) >= max(50, self.out_dim * 2):
                    X = np.stack(self._pca_buffer, axis=0)
                    self._pca.fit(X)
                    self._pca_fitted = True
                    self._pca_buffer = []
                else:
                    # fallback to linear project until PCA ready
                    out = x[:self.out_dim].copy()
                    if out.shape[0] < self.out_dim:
                        out = np.pad(out, (0, self.out_dim - out.shape[0]))
                    return out
            return self._pca.transform(x.reshape(1, -1)).reshape(-1)
        else:
            return x.dot(self.W) + self.b

class TemporalFabric:
    def __init__(self, emb_dim: int, cfg: Optional[TemporalConfig] = None):
        self.emb_dim = emb_dim
        self.cfg = cfg or TemporalConfig()
        self.short_buffer = deque(maxlen=self.cfg.N_short)  # stores raw embeddings (numpy arrays)
        self.mid_segments: List[Segment] = []
        self.long_store: Dict[int, Dict[str, Any]] = {}
        self.mod_state = dict(self.cfg.mod_init)
        self.energy = self.cfg.energy_max
        self.step = 0
        self._last_dream = 0

        # emotion banks: vector of length len(taus)
        self.current_emotion = np.zeros((len(self.cfg.emotion_taus),), dtype=float)

        # conv lite + compressor
        self.conv = DilatedConvLite(emb_dim, self.cfg)
        input_dim = emb_dim * len(self.cfg.dilation_profile)
        self.compressor = LinearCompressor(input_dim, self.cfg.mid_emb_dim, use_pca=False)

    # ----------------- emotion / neuromodulator -----------------
    def update_emotion(self, emotion_vector: np.ndarray, dt: float = 1.0):
        """Update multi-timescale emotion banks (elementwise)."""
        taus = self.cfg.emotion_taus
        decay = np.exp(-dt / taus)
        self.current_emotion = self.current_emotion * decay + np.asarray(emotion_vector, dtype=float) * (1.0 - decay)

    def _update_modulators_from_som(self, som):
        """
        Heuristic update: uses som.failure_log length, recent quantization error
        (if available), and BMU utilization to update modulators.
        This is intentionally simple and transparent.
        """
        # dopamine: scaled by novelty indicator (use failure_log length as novelty proxy)
        try:
            novelty_proxy = len(som.failure_log) if hasattr(som, "failure_log") else 0
        except Exception:
            novelty_proxy = 0
        self.mod_state["dopamine"] = 1.0 + 0.1 * np.tanh(novelty_proxy)

        # serotonin: inverse volatility: use variance of som.quantization_errors if available
        try:
            qe = np.array(getattr(som, "quantization_errors", [])[-50:], dtype=float)
            volatility = float(np.var(qe)) if qe.size > 1 else 0.0
            self.mod_state["serotonin"] = 1.0 / (1.0 + 0.5 * volatility)
        except Exception:
            self.mod_state["serotonin"] = 1.0

        # norepi: increase with recent prediction / quantization error (alertness)
        try:
            recent_qe = float(np.mean(getattr(som, "quantization_errors", [])[-5:])) if getattr(som, "quantization_errors", None) else 0.0
            self.mod_state["norepi"] = 1.0 + 0.5 * np.tanh(recent_qe)
        except Exception:
            self.mod_state["norepi"] = 1.0

        # oxytocin: if som exposes social tag rate, otherwise small baseline
        self.mod_state["oxytocin"] = getattr(som, "social_tag_rate", 0.0) if hasattr(som, "social_tag_rate") else 0.1

    # ----------------- main streaming step -----------------
    def step_forward(self, embedding: np.ndarray, raw_emotion_signal: Optional[np.ndarray] = None, som=None):
        """
        Main streaming entry point.
        embedding: 1D numpy array length emb_dim.
        raw_emotion_signal: 1D array length equal to len(emotion_taus) or scalar broadcastable.
        som: optional SelfOrganizingMap instance to read novelty / failure logs from.
        """
        self.step += 1
        emb = np.asarray(embedding, dtype=np.float16)
        if emb.shape[0] != self.emb_dim:
            raise ValueError(f"embedding dim {emb.shape[0]} != expected {self.emb_dim}")

        self.short_buffer.append(emb)

        # update emotion
        if raw_emotion_signal is None:
            raw_emotion_signal = np.zeros_like(self.current_emotion)
        else:
            raw_emotion_signal = np.asarray(raw_emotion_signal, dtype=float)
            if raw_emotion_signal.shape != self.current_emotion.shape:
                # try to broadcast or pad/truncate
                if raw_emotion_signal.size == 1:
                    raw_emotion_signal = np.ones_like(self.current_emotion) * float(raw_emotion_signal)
                else:
                    tmp = np.zeros_like(self.current_emotion)
                    tmp[:min(tmp.size, raw_emotion_signal.size)] = raw_emotion_signal[:tmp.size]
                    raw_emotion_signal = tmp
        self.update_emotion(raw_emotion_signal, dt=1.0)

        # update modulators from som if provided
        if som is not None:
            try:
                self._update_modulators_from_som(som)
            except Exception:
                pass

        # segment boundary
        if self.step % self.cfg.N_segment_steps == 0:
            self._close_segment_and_compress()

        # energy recovery
        if self.energy < self.cfg.energy_max:
            self.energy = min(self.cfg.energy_max, self.energy + self.cfg.energy_recover_rate)

        # dream consolidation trigger
        if (self.step - self._last_dream) > self.cfg.dream_interval_steps:
            self.run_dream_phase()
            self._last_dream = self.step

    # ----------------- segment close & compression -----------------
    def _close_segment_and_compress(self):
        # gather last N_segment_steps from short_buffer (if available)
        seg_len = min(self.cfg.N_segment_steps, len(self.short_buffer))
        if seg_len == 0:
            return
        seq = list(self.short_buffer)[-seg_len:]
        # conv-lite last-step representation
        conv_out = self.conv.forward_last(seq)  # length = emb_dim * num_dilations
        mid_emb = self.compressor.compress(conv_out).astype(np.float16)

        # emotion signature: snapshot copy of current_emotion
        emotion_sig = self.current_emotion.copy()

        # novelty proxy: variance across segment embeddings (cheap)
        seg_np = np.stack(seq, axis=0)
        novelty = float(np.var(seg_np))

        ts = time.time()
        seg = Segment(embedding=mid_emb, emotion_sig=emotion_sig, novelty=novelty, timestamp=ts, meta={"len": seg_len})
        self.mid_segments.append(seg)

        # energy accounting
        effective_write_cost = self.cfg.write_cost / max(1e-6, self.mod_state.get("serotonin", 1.0))
        if self.energy >= effective_write_cost:
            self.energy -= effective_write_cost
        else:
            # create ghost pointer instead
            self._make_ghost(seg)
            return

        # optionally apply immediate retention policy to keep mid size sane
        if len(self.mid_segments) > int(self.cfg.mid_soft_cap * (1 + 0.5 * (1 - self.mod_state.get("dopamine", 1.0)))):
            # score and prune bottom fraction
            self._prune_mid_segments(frac=0.2)

    def _make_ghost(self, segment: Segment):
        key_hash = hash((segment.timestamp, int(segment.novelty * 1e6)))
        self.long_store[key_hash] = {"ghost": True, "emotion": segment.emotion_sig, "anchors": [segment.embedding[:2].tolist()]}
        # don't append to mid_segments

    # ----------------- retention scoring & pruning -----------------
    def score_segment_for_retention(self, segment: Segment, now: Optional[float] = None, prediction_gain_estimate: float = 0.0):
        if now is None:
            now = time.time()
        dt = now - segment.timestamp
        recency_decay = np.exp(-dt / max(1.0, self.cfg.recency_tau))
        emotion_intensity = float(np.linalg.norm(segment.emotion_sig))
        # normalize novelty and emotion lightly (small running normalizer could be added later)
        n = segment.novelty
        score = (self.cfg.w_n * n + self.cfg.w_e * emotion_intensity + self.cfg.w_r * recency_decay + self.cfg.w_p * prediction_gain_estimate)
        return float(score)

    def _prune_mid_segments(self, frac: float = 0.2):
        if not self.mid_segments:
            return
        now = time.time()
        scores = [(i, self.score_segment_for_retention(s, now)) for i, s in enumerate(self.mid_segments)]
        scores.sort(key=lambda x: x[1])  # ascending, low score first
        prune_count = max(1, int(len(scores) * frac))
        prune_idx = set(i for i, _ in scores[:prune_count])
        for i in sorted(prune_idx, reverse=True):
            s = self.mid_segments[i]
            self._make_ghost(s)
            del self.mid_segments[i]

    # ----------------- dream / consolidation -----------------
    def run_dream_phase(self):
        # simple consolidation: if mid store is big, prune low-score items and refill energy
        if len(self.mid_segments) > self.cfg.mid_soft_cap:
            self._prune_mid_segments(frac=0.25)
        # refresh some mid segments by re-projecting them (no-op here), and refill energy proportionally
        self.energy = min(self.cfg.energy_max, self.energy + 0.5 * self.cfg.energy_max)

    # ----------------- retrieval -----------------
    def retrieve_similar(self, query_emb: np.ndarray, top_k: int = 5):
        """Naive linear scan retrieval over mid_segments. Returns top_k segments."""
        if len(self.mid_segments) == 0:
            return []
        q = np.asarray(query_emb, dtype=float)
        qn = q / (np.linalg.norm(q) + 1e-9)
        scored = []
        for s in self.mid_segments:
            se = s.embedding
            se_n = se / (np.linalg.norm(se) + 1e-9)
            sim = float(np.dot(qn, se_n))
            # emotion match term
            em = 0.0
            if np.linalg.norm(s.emotion_sig) > 1e-9:
                em = float(np.dot(self.current_emotion, s.emotion_sig) / (np.linalg.norm(s.emotion_sig) + 1e-9))
            score = sim + 0.5 * em + 0.1 * s.novelty
            scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:top_k]]

    # ----------------- diagnostics -----------------
    def stats(self):
        return {
            "step": int(self.step),
            "mid_count": len(self.mid_segments),
            "long_count": len(self.long_store),
            "energy": float(self.energy),
            "mod_state": dict(self.mod_state)
        }

# Lightweight integration helper
def attach_to_som_and_feed(tf: TemporalFabric, som, stream_embeddings: List[np.ndarray], emotion_signals: Optional[List[np.ndarray]] = None):
    """
    Feed a list of embeddings into tf and the som together. This is a convenience to align signals.
    """
    emotion_signals = emotion_signals or [np.zeros_like(tf.current_emotion) for _ in stream_embeddings]
    for emb, emo in zip(stream_embeddings, emotion_signals):
        # send to SOM first (if it has process_input)
        try:
            som.process_input(np.asarray(emb, dtype=np.float32))
        except Exception:
            pass
        tf.step_forward(emb, raw_emotion_signal=emo, som=som)


# How to drop in & quick examples

#1. Save the code above as `temporal.py` in the same folder as your `nn.py` and `som.py`.
#2. Basic usage example (Python REPL or a script):


from som import SelfOrganizingMap
from temporal import TemporalFabric
import numpy as np

som = SelfOrganizingMap(map_size=(17,17), input_dim=80)
tf = TemporalFabric(emb_dim=80)   # if your embeddings are size 80 (SOM default)

# feed 1000 random steps
for i in range(1000):
    emb = np.random.randn(80).astype(np.float32)
    # Real emotion signals come from Emotion.py processing actual stimuli
    # emo = emotion_module.get_current_state()  # Replace with real emotion data
    som.process_input(emb)          # update som
    tf.step_forward(emb, som=som)

print(tf.stats())
# retrieve using last mid embedding if any
if tf.mid_segments:
    q = tf.mid_segments[-1].embedding
    found = tf.retrieve_similar(q, top_k=3)
    print("retrieved", len(found))

# Notes, gotchas & next micro-steps (pick one)

"""
* This implementation intentionally keeps things simple and pure NumPy
 so it integrates smoothly with your `nn.py` and `som.py`.
* The dilated conv is a light approximation (learnable kernels per dilation).
 It's slower than optimized libs but fully transparent and editable.
* Mid compression uses a linear projection by default. If you want 
PCA-powered compressor, enable `use_pca=True` in `LinearCompressor` and ensure `scikit-learn` 
is installed.
* I did not wire a full prediction-error model; novelty uses 
segment variance for now. You already have SOM quantization errors — those are read into 
neuromodulator heuristics.
* Next micro-steps I can implement right now (pick one, I’ll paste code):

  1. Replace linear compressor with a tiny trainable NumPy autoencoder + training loop 
  (online-friendly).
  2. Add ANN retrieval wiring using hnswlib (if you want speed and can install it).
  3. Implement the backward micro-pass over compressed chunk and stable XOR key generation.
  4. Add stronger neuromodulator update rules tied to SOM BMU coords and prediction error signals.
  """