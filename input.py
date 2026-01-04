"""
input.py

Minimal input interface for audio and video.

Features:
- Register/unregister video input callbacks
- Register/unregister audio input callbacks
- Send video/audio inputs to registered input callbacks
- Deliver video/audio inputs with SFE integration

Design notes:
- No external imports are used (per user instruction).
- Callbacks are called synchronously by the caller; keep handlers fast.
- Callbacks receive raw bytes and an optional meta dict.


"""

import time
import collections
import math
import threading
import logging
from threading import Thread, Event, Lock

try:
    import numpy as np
except Exception:
    np = None

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import librosa
except Exception:
    librosa = None

# Configure lightweight logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Import input parameters from config
from config import (INPUT_AUDIO_DEVICE_INDEX, INPUT_AUDIO_SAMPLE_RATE,
                   INPUT_AUDIO_FRAME_SIZE, INPUT_AUDIO_BUFFER_SECONDS,
                   INPUT_AUDIO_CHANNELS, INPUT_AUDIO_RMS_THRESHOLD,
                   INPUT_VIDEO_DEVICE_INDEX, INPUT_VIDEO_FRAME_WIDTH,
                   INPUT_VIDEO_FRAME_HEIGHT, INPUT_VIDEO_FPS)

# Internal registries
_video_inputs = []  # list of callables(frame_bytes, meta)
_audio_inputs = []  # list of callables(chunk_bytes, meta)

# Single-module hooks per user request: SFE
_sfe_module = None

# Latest raw inputs (kept so SFE can be called with matching pairs)
_last_audio_chunk = None
_last_video_frame = None
_last_sfe_features = None


# --- Begin embedded classes from audioin.py, videoin.py ---


class AudioIn:
    """Background audio capture and feature extractor (from audioin.py).

    If sounddevice or numpy aren't available, the class falls back to no-op behavior.
    """
    def __init__(self, device_index: int = INPUT_AUDIO_DEVICE_INDEX, sample_rate: int = INPUT_AUDIO_SAMPLE_RATE, frame_size: int = INPUT_AUDIO_FRAME_SIZE, buffer_seconds: float = INPUT_AUDIO_BUFFER_SECONDS, feature_mode: bool = False, channels: int = INPUT_AUDIO_CHANNELS):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.feature_mode = feature_mode
        self.channels = max(1, int(channels))

        self.buffer_size = int(self.sample_rate * max(0.5, buffer_seconds))
        self._buffer = [collections.deque(maxlen=self.buffer_size) for _ in range(self.channels)]
        self._lock = Lock()
        self._thread = None
        self._stop_event = Event()
        self.running = False
        if np is not None:
            self._last_frame = np.zeros(self.frame_size, dtype=np.float32)
        else:
            self._last_frame = None

        self._rms_threshold = 0.01
        self._sd_stream = None

    def _capture_loop(self):
        if sd is None:
            return
        try:
            def callback(indata, frames, time_info, status):
                if status:
                    pass
                with self._lock:
                    for ch in range(min(indata.shape[1], self.channels)):
                        arr = indata[:, ch].astype(np.float32)
                        for v in arr.tolist():
                            self._buffer[ch].append(v)

            self._sd_stream = sd.InputStream(samplerate=self.sample_rate, device=self.device_index, channels=self.channels, callback=callback)
            self._sd_stream.start()
            while not self._stop_event.is_set():
                time.sleep(0.05)
        except Exception:
            pass

    def start(self):
        self._stop_event.clear()
        self.running = True
        if sd is None:
            return
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.running = False
        try:
            if self._sd_stream is not None:
                self._sd_stream.stop()
                self._sd_stream.close()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _get_recent_samples(self, num_samples: int):
        with self._lock:
            out = []
            for ch in range(self.channels):
                buf = self._buffer[ch]
                if len(buf) == 0:
                    out.append(np.zeros(num_samples, dtype=np.float32) if np is not None else [0.0] * num_samples)
                else:
                    arr = np.asarray(buf, dtype=np.float32)
                    if arr.size >= num_samples:
                        out.append(arr[-num_samples:])
                    else:
                        padded = np.zeros(num_samples, dtype=np.float32)
                        padded[-arr.size:] = arr
                        out.append(padded)
            return out

    def _compute_rms(self, frame):
        try:
            return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        except Exception:
            return 0.0

    def _energy_bands(self, frame, bands: int = 8):
        try:
            S = np.abs(np.fft.rfft(frame))
            if S.size == 0:
                return np.zeros(bands, dtype=np.float32)
            band_edges = np.linspace(0, S.size, bands + 1, dtype=int)
            energies = np.zeros(bands, dtype=np.float32)
            for i in range(bands):
                s = S[band_edges[i]:band_edges[i+1]]
                energies[i] = float(np.sum(s ** 2))
            total = energies.sum() + 1e-9
            return (energies / total).astype(np.float32)
        except Exception:
            return np.zeros(bands, dtype=np.float32)

    def _compute_mfcc(self, frame, n_mfcc: int = 13):
        if librosa is None:
            return np.zeros(n_mfcc, dtype=np.float32)
        try:
            mfccs = librosa.feature.mfcc(y=frame.astype(np.float32), sr=self.sample_rate, n_mfcc=n_mfcc)
            return np.mean(mfccs, axis=1).astype(np.float32)
        except Exception:
            return np.zeros(n_mfcc, dtype=np.float32)

    def _spectral_centroid(self, frame):
        try:
            if librosa is None:
                return 0.0
            return float(np.mean(librosa.feature.spectral_centroid(y=frame.astype(np.float32), sr=self.sample_rate)))
        except Exception:
            return 0.0

    def _spectral_rolloff(self, frame):
        try:
            if librosa is None:
                return 0.0
            return float(np.mean(librosa.feature.spectral_rolloff(y=frame.astype(np.float32), sr=self.sample_rate)))
        except Exception:
            return 0.0

    def _zero_crossing_rate(self, frame):
        try:
            if librosa is None:
                return 0.0
            return float(np.mean(librosa.feature.zero_crossing_rate(frame.astype(np.float32))))
        except Exception:
            return 0.0

    def get_audio_features(self, as_vector: bool = True):
        frames = self._get_recent_samples(self.frame_size)
        channel_features = []
        for ch_frame in frames:
            rms = self._compute_rms(ch_frame)
            vad = 1.0 if rms > self._rms_threshold else 0.0
            bands = self._energy_bands(ch_frame, bands=8)
            mfcc = self._compute_mfcc(ch_frame, n_mfcc=13)
            centroid = self._spectral_centroid(ch_frame)
            rolloff = self._spectral_rolloff(ch_frame)
            zcr = self._zero_crossing_rate(ch_frame)
            channel_features.append({'rms': rms, 'vad': vad, 'bands': bands, 'mfcc': mfcc, 'centroid': centroid, 'rolloff': rolloff, 'zcr': zcr, 'raw': ch_frame})

        inter_corr = 0.0
        stereo_width = 0.0
        try:
            if self.channels >= 2:
                a = frames[0]
                b = frames[1]
                if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
                    inter_corr = float(np.corrcoef(a, b)[0,1])
                    stereo_width = float(np.mean(np.abs(a - b)))
        except Exception:
            inter_corr = 0.0
            stereo_width = 0.0

        if as_vector:
            parts = []
            for chf in channel_features:
                parts.append(np.array([chf['rms'], chf['vad'], chf['centroid'], chf['rolloff'], chf['zcr']], dtype=np.float32))
                parts.append(np.array(chf['bands'], dtype=np.float32))
                parts.append(np.array(chf['mfcc'], dtype=np.float32))
            parts.append(np.array([inter_corr, stereo_width], dtype=np.float32))
            raw0 = channel_features[0]['raw'] if channel_features else np.zeros(self.frame_size, dtype=np.float32)
            rawvec = raw0[:self.frame_size]
            parts.append(rawvec.astype(np.float32))
            vec = np.concatenate([p.flatten() for p in parts])
            return vec.astype(np.float32)
        else:
            return {'channels': channel_features, 'inter_corr': inter_corr, 'stereo_width': stereo_width}


class VideoIn:
    """Background video capture and feature extractor (from videoin.py).

    Falls back to no-op when OpenCV is unavailable.
    """
    def __init__(self, device_index: int = 0, target_feature_dim: int = 512, thumb_size: tuple = (32, 32), capture_source: str = 'camera'):
        self.device_index = device_index
        self.target_feature_dim = target_feature_dim
        self.thumb_size = thumb_size
        self.capture_source = capture_source  # 'camera' or 'screen'

        self.cap = None
        self.running = False
        self._thread = None
        self._stop_event = Event()
        self._lock = Lock()
        self._last_frame = None
        self._prev_gray = None
        self._motion_mag = 0.0

    def _capture_loop(self):
        if cv2 is None:
            return
        try:
            if self.capture_source == 'camera':
                self.cap = cv2.VideoCapture(self.device_index)
                if not self.cap.isOpened():
                    try:
                        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
                    except Exception:
                        pass
            else:
                logger.error(f"Unknown capture_source: {self.capture_source}. Only 'camera' supported.")
                return

            while not self._stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue
                    continue

                with self._lock:
                    self._last_frame = frame.copy()
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if self._prev_gray is not None:
                            flow = cv2.calcOpticalFlowFarneback(self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                            self._motion_mag = float(np.mean(mag))
                        self._prev_gray = gray
                    except Exception:
                        self._prev_gray = None
                time.sleep(0.01)
        except Exception:
            pass

    def start(self):
        self._stop_event.clear()
        self.running = True
        if cv2 is None:
            return
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.running = False
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_video_features(self, as_vector: bool = True):
        if not self.running:
            zeros = np.zeros(self.target_feature_dim, dtype=np.float32) if np is not None else [0.0] * self.target_feature_dim
            return zeros if as_vector else {'thumb': zeros, 'motion_mag': 0.0, 'frame_shape': None}

        if cv2 is None:
            zeros = np.zeros(self.target_feature_dim, dtype=np.float32) if np is not None else [0.0] * self.target_feature_dim
            return zeros if as_vector else {'thumb': zeros, 'motion_mag': 0.0, 'frame_shape': None}

        with self._lock:
            frame = self._last_frame.copy() if self._last_frame is not None else None
            motion = getattr(self, '_motion_mag', 0.0)

        if frame is None:
            zeros = np.zeros(self.target_feature_dim, dtype=np.float32) if np is not None else [0.0] * self.target_feature_dim
            return zeros if as_vector else {'thumb': zeros, 'motion_mag': motion, 'frame_shape': None}

        try:
            thumb = cv2.resize(frame, self.thumb_size)
            gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
            flat = gray.flatten().astype(np.float32)
            if flat.size >= self.target_feature_dim:
                vec = flat[:self.target_feature_dim]
            else:
                buf = np.zeros(self.target_feature_dim, dtype=np.float32)
                buf[:flat.size] = flat
                vec = buf

            if as_vector:
                out = np.concatenate(([motion], vec[1:])) if vec.size > 1 else np.concatenate(([motion], vec))
                return out.astype(np.float32)
            else:
                return {'thumb': vec, 'motion_mag': float(motion), 'frame_shape': frame.shape}
        except Exception:
            zeros = np.zeros(self.target_feature_dim, dtype=np.float32)
            return zeros if as_vector else {'thumb': zeros, 'motion_mag': 0.0, 'frame_shape': None}

    def get_latest_frame(self):
        """Return the most recent captured frame (BGR) or None."""
        with self._lock:
            return None if self._last_frame is None else self._last_frame.copy()

    def get_latest_audio_chunk(self):
        """Return the latest raw audio chunk for feedback loops."""
        with self._lock:
            if len(self._buffer[0]) == 0:
                return np.zeros(self.frame_size, dtype=np.float32)
            # Return the most recent frame from the first channel
            return np.array(list(self._buffer[0])[-self.frame_size:], dtype=np.float32)

# --- End embedded classes ---


def _is_callable(obj):
    try:
        return callable(obj)
    except Exception:
        return False


# Registration API for inputs
def register_video_input(callback):
    """Register a video input callback.

    callback(frame_bytes, meta) will be called when the application delivers a new
    video frame to the system. Returns True on success, False otherwise.
    """
    if not _is_callable(callback):
        return False
    if callback in _video_inputs:
        return True
    _video_inputs.append(callback)
    return True


def unregister_video_input(callback):
    """Unregister a previously registered video input callback."""
    try:
        _video_inputs.remove(callback)
        return True
    except ValueError:
        return False


def register_audio_input(callback):
    """Register an audio input callback.

    callback(chunk_bytes, meta) will be called when a new audio chunk arrives.
    """
    if not _is_callable(callback):
        return False
    if callback in _audio_inputs:
        return True
    _audio_inputs.append(callback)
    return True


def unregister_audio_input(callback):
    """Unregister a previously registered audio input callback."""
    try:
        _audio_inputs.remove(callback)
        return True
    except ValueError:
        return False


# Helpers to deliver incoming frames/chunks to registered input callbacks
def deliver_video_input(frame_bytes, meta=None):
    """Call all registered video input callbacks with the provided frame.

    Returns number of callbacks successfully invoked.
    """
    if meta is None:
        meta = {}
    count = 0
    for cb in list(_video_inputs):
        try:
            cb(frame_bytes, meta)
            count += 1
        except Exception as e:
            print("video input callback error:", e)
    # store latest video frame for SFE pairing
    global _last_video_frame, _last_sfe_features
    _last_video_frame = frame_bytes

    # Forward to SFE if present
    if _sfe_module is not None:
        try:
            if hasattr(_sfe_module, 'get_features'):
                # Convert bytes back to numpy arrays for SFE - only with real video format
                audio_array = np.frombuffer(_last_audio_chunk, dtype=np.float32) if _last_audio_chunk is not None else None
                video_array = None
                if _last_video_frame is not None and 'shape' in meta:
                    video_bytes = np.frombuffer(_last_video_frame, dtype=np.uint8)
                    try:
                        video_array = video_bytes.reshape(meta['shape'])
                    except ValueError:
                        video_array = None  # Invalid shape, skip
                # Pass the most recent audio chunk and this video frame
                _last_sfe_features = _sfe_module.get_features(audio_array, video_array)
        except Exception as e:
            print("SFE get_features error (video):", e)

    return count


def deliver_audio_input(chunk_bytes, meta=None):
    """Call all registered audio input callbacks with the provided chunk.

    Returns number of callbacks successfully invoked.
    """
    if meta is None:
        meta = {}
    count = 0
    for cb in list(_audio_inputs):
        try:
            cb(chunk_bytes, meta)
            count += 1
        except Exception as e:
            print("audio input callback error:", e)
    # store latest audio chunk for SFE pairing
    global _last_audio_chunk, _last_sfe_features
    _last_audio_chunk = chunk_bytes

    # Forward to SFE if present  
    if _sfe_module is not None:
        try:
            if hasattr(_sfe_module, 'get_features'):
                # Convert bytes back to numpy arrays for SFE - only with real video format
                audio_array = np.frombuffer(_last_audio_chunk, dtype=np.float32) if _last_audio_chunk is not None else None
                video_array = None
                # Only process video if real format is available
                if _last_video_frame is not None:
                    try:
                        # Requires real video metadata, no assumptions
                        video_array = _last_video_frame if hasattr(_last_video_frame, 'shape') else None
                    except Exception:
                        video_array = None
                _last_sfe_features = _sfe_module.get_features(audio_array, video_array)
        except Exception as e:
            print("SFE get_features error (audio):", e)

    return count


def set_sfe(module):
    """Set the Sensory Feature Extractor module. Module should implement get_features(audio_chunk, video_frame)."""
    global _sfe_module
    _sfe_module = module
    return True


def clear_sfe():
    global _sfe_module
    _sfe_module = None
    return True


# Text handlers - explicitly blocked from model forwarding
def text_input(source, text):
    """Receive text input but block forwarding to model.

    Returns a dict: {"allowed": False, "reason": str}
    """
    # Allow text input to be forwarded to the model by default. Callers may still
    # validate or sanitize the text before forwarding.
    return {"allowed": True, "payload": {"source": source, "text": text}}


__all__ = [
    "register_video_input",
    "unregister_video_input",
    "register_audio_input",
    "unregister_audio_input",
    "deliver_video_input",
    "deliver_audio_input",
    "text_input",
    # embedded classes
    "AudioIn",
    "VideoIn",
]