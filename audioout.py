"""
audioout.py

Audio output playback using sounddevice.

Provides a queue-based output with start/stop lifecycle and play_chunk/play_audio wrapper.
"""

import queue
import logging

try:
    import numpy as np
except Exception:
    np = None

try:
    import sounddevice as sd
except Exception:
    sd = None

# Configure lightweight logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AudioOut:
    """Audio output playback using sounddevice.

    Provides a queue-based output with start/stop lifecycle and play_chunk/play_audio wrapper.
    """
    def __init__(self, device_index: int = None, sample_rate: int = 44100, chunk_duration: float = 0.1):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.output_queue = queue.Queue(maxsize=10)
        self.running = False
        self.stream = None
        logger.info(f"AudioOut initialized. Sample Rate: {sample_rate}, Chunk Duration: {chunk_duration}s.")
        logger.info(f"Expected input audio_chunk dtype: np.float64, range: -1.0 to 1.0.")

    def _output_callback(self, outdata, frames, time_info, status):
        if status:
            logger.warning(f"AudioOut Warning during callback: {status}")
        try:
            data = self.output_queue.get_nowait()
            data = data.copy().astype(np.float32)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if len(data) >= len(outdata):
                outdata[:] = data[:len(outdata)]
            else:
                outdata[:len(data)] = data
                outdata[len(data):].fill(0)

        except queue.Empty:
            outdata.fill(0)
        except Exception as e:
            logger.error(f"Error in AudioOut callback: {e}")
            outdata.fill(0)

    def start(self):
        if self.running:
            logger.warning("AudioOut stream is already running.")
            return
        try:
            if sd is None:
                logger.error("sounddevice not available; cannot start audio output stream.")
                self.running = False
                return
            self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype='float32', device=self.device_index, blocksize=self.chunk_size, callback=self._output_callback)
            self.stream.start()
            self.running = True
            logger.info(f"AudioOut stream started on device index: {self.device_index if self.device_index is not None else 'Default'}.")
        except Exception as e:
            logger.error(f"FAILED TO START AUDIO OUTPUT STREAM: {e}. Check device index or speaker access.")
            self.running = False

    def stop(self):
        if self.stream is not None and getattr(self.stream, 'is_active', False):
            self.running = False
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass
            logger.info("AudioOut stream stopped and queue cleared.")
        elif not self.running:
            logger.info("AudioOut stream is not active or already stopped.")

    def play_chunk(self, audio_chunk):
        if not self.running:
            logger.warning("AudioOut stream is not running. Cannot play audio.")
            return
        if np is None or not isinstance(audio_chunk, np.ndarray) or audio_chunk.dtype != np.float64:
            try:
                dtype = getattr(audio_chunk, 'dtype', None)
            except Exception:
                dtype = None
            logger.error(f"AudioOut.play_chunk expected np.ndarray (float64), got {type(audio_chunk)} with dtype {dtype}. Skipping.")
            return
        try:
            self.output_queue.put(audio_chunk)
        except queue.Full:
            logger.warning("AudioOut queue is full. Dropping audio chunk.")
        except Exception as e:
            logger.error(f"Error adding audio chunk to queue: {e}")

    def play_audio(self, audio_chunk):
        self.play_chunk(audio_chunk)
