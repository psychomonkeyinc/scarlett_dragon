# vocalsynth.py

import os
import numpy as np
import logging
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
# import inout  # Module not available - output disabled
import config
from config import (VOCALSYNTH_SAMPLE_RATE, VOCALSYNTH_BUFFER_SIZE_SAMPLES,
                   VOCALSYNTH_DEFAULT_F0_HZ, VOCALSYNTH_DEFAULT_VIBRATO_RATE,
                   VOCALSYNTH_DEFAULT_VIBRATO_EXTENT, VOCALSYNTH_DEFAULT_AMPLITUDE,
                   VOCALSYNTH_DEFAULT_VOICING_MIX, VOCALSYNTH_DEFAULT_SPECTRAL_TILT,
                   VOCALSYNTH_DEFAULT_BREATHINESS, VOCALSYNTH_DEFAULT_F1_HZ,
                   VOCALSYNTH_DEFAULT_F2_HZ, VOCALSYNTH_DEFAULT_F3_HZ,
                   VOCALSYNTH_DEFAULT_F4_HZ, VOCALSYNTH_DEFAULT_BW1_HZ,
                   VOCALSYNTH_DEFAULT_BW2_HZ, VOCALSYNTH_DEFAULT_BW3_HZ,
                   VOCALSYNTH_DEFAULT_BW4_HZ, VOCALSYNTH_DEFAULT_TENSION,
                   VOCALSYNTH_DEFAULT_ROUGHNESS)

# Single consolidated state file for all modules
GLOBAL_STATE_FILE = Path(__file__).resolve().parent / "master" / "master_model_state.pkl"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# Biquad Filter (Helper for Formant Synthesis)
# A direct form 2 transposed biquad filter implementation for high precision.
# ==============================================================================

class BiquadFilter:
    """
    Implements a biquad (second-order) IIR filter. Used for modeling formants.
    Coefficients: b0, b1, b2 (numerator) and a0, a1, a2 (denominator).
    """
    def __init__(self, b_coeffs: np.ndarray, a_coeffs: np.ndarray):
        # Ensure coefficients use dynamic precision from config
        self.b = b_coeffs.astype(config.FLOATX) # [b0, b1, b2]
        self.a = a_coeffs.astype(config.FLOATX) # [a0, a1, a2] (a0 is usually 1.0)

        # Internal state variables (for direct form 2 transposed)
        self.z1 = config.FLOATX(0.0)
        self.z2 = config.FLOATX(0.0)

        # Normalize by a0 if it's not 1.0 (though for resonant filters, it usually is)
        if self.a[0] != 1.0:
            self.b /= self.a[0]
            self.a /= self.a[0]

    def process_sample(self, x: config.FLOATX) -> config.FLOATX:
        """Processes a single audio sample."""
        # y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        # Direct form 2 transposed
        y = self.b[0] * x + self.z1
        self.z1 = self.b[1] * x - self.a[1] * y + self.z2
        self.z2 = self.b[2] * x - self.a[2] * y
        return y

    def reset_state(self):
        """Resets the internal filter state."""
        self.z1 = config.FLOATX(0.0)
        self.z2 = config.FLOATX(0.0)

# ==============================================================================
# VocalSynth: Lillith's Internal Vocal Apparatus
# ==============================================================================

class VocalSynth:
    """
    Lillith's Internal Vocal Apparatus: Pure NumPy implementation of a simplified
    source-filter model for speech synthesis, using np.float16 precision.
    """
    def __init__(self, sample_rate: int = VOCALSYNTH_SAMPLE_RATE, buffer_size_samples: int = VOCALSYNTH_BUFFER_SIZE_SAMPLES): # 0.1 second buffer
        self.sample_rate = sample_rate
        self.buffer_size_samples = buffer_size_samples
        self.current_time = config.FLOATX(0.0) # Tracks time for waveform generation
        
        # --- Default/Initial Synthesis Parameters (will be controlled by Output.py) ---
        self.params = {
            # Pitch & Timing (Source Control)
            'F0_hz': config.FLOATX(VOCALSYNTH_DEFAULT_F0_HZ),          # Fundamental Frequency (pitch)
            'Vibrato_Rate': config.FLOATX(VOCALSYNTH_DEFAULT_VIBRATO_RATE),     # Hz
            'Vibrato_Extent': config.FLOATX(VOCALSYNTH_DEFAULT_VIBRATO_EXTENT),   # Hz deviation
            # Loudness & Voice Quality (Source & Mix Control)
            'Amplitude': config.FLOATX(VOCALSYNTH_DEFAULT_AMPLITUDE),        # Overall loudness (0-1)
            'Voicing_Mix': config.FLOATX(VOCALSYNTH_DEFAULT_VOICING_MIX),      # 1.0 = fully voiced, 0.0 = fully unvoiced
            'Spectral_Tilt': config.FLOATX(VOCALSYNTH_DEFAULT_SPECTRAL_TILT),    # Affects brightness (e.g., -1.0 to 1.0)
            'Breathiness_Amount': config.FLOATX(VOCALSYNTH_DEFAULT_BREATHINESS), # (0-1)
            # Vocal Tract Shape (Filter Control - Formants)
            'F1_hz': config.FLOATX(VOCALSYNTH_DEFAULT_F1_HZ),          # Formant 1 frequency (e.g., for 'ah' vowel)
            'F2_hz': config.FLOATX(VOCALSYNTH_DEFAULT_F2_HZ),         # Formant 2 frequency
            'F3_hz': config.FLOATX(VOCALSYNTH_DEFAULT_F3_HZ),         # Formant 3 frequency
            'F4_hz': config.FLOATX(VOCALSYNTH_DEFAULT_F4_HZ),         # Formant 4 frequency
            'BW1_hz': config.FLOATX(VOCALSYNTH_DEFAULT_BW1_HZ),          # Formant 1 bandwidth
            'BW2_hz': config.FLOATX(VOCALSYNTH_DEFAULT_BW2_HZ),          # Formant 2 bandwidth
            'BW3_hz': config.FLOATX(VOCALSYNTH_DEFAULT_BW3_HZ),         # Formant 3 bandwidth
            'BW4_hz': config.FLOATX(VOCALSYNTH_DEFAULT_BW4_HZ),         # Formant 4 bandwidth
            # Dynamic Expression (Overall Shaping)
            'Tension_Param': config.FLOATX(VOCALSYNTH_DEFAULT_TENSION),    # (e.g., -1.0 to 1.0)
            'Roughness_Param': config.FLOATX(VOCALSYNTH_DEFAULT_ROUGHNESS),  # (0-1)
        }
        # Ensure all parameters use dynamic precision
        for k in self.params:
            self.params[k] = config.FLOATX(self.params[k])

        # --- Internal State for Source Model ---
        self._phase = config.FLOATX(0.0)
        self._last_f0 = self.params['F0_hz'] # For smooth pitch transitions
        self._glottal_source_state = config.FLOATX(0.0) # For simple pulse train

        # --- Internal State for Filter Model (Formant Filters) ---
        self.formant_filters = []
        self._initialize_formant_filters()
        
        logger.info(f"VocalSynth initialized with sample_rate={sample_rate}, buffer_size={buffer_size_samples}.")

    def _initialize_formant_filters(self):
        """Initializes/updates the biquad filters for formants."""
        self.formant_filters = []
        formant_indices = [1, 2, 3, 4] # For F1, F2, F3, F4
        
        for i in formant_indices:
            F = self.params[f'F{i}_hz']
            BW = self.params[f'BW{i}_hz']
            
            # Calculate filter coefficients for a resonant peak (biquad filter)
            # Based on standard digital filter design for resonant peaks (e.g., peaking EQ)
            # Or simpler direct formants
            
            # Pole-zero filter coefficients for a single formant
            # r = np.exp(-np.pi * BW / self.sample_rate) # Radius
            # theta = 2 * np.pi * F / self.sample_rate   # Angle
            # a1 = -2 * r * np.cos(theta)
            # a2 = r * r
            # b0 = 1.0
            # b1 = 0.0
            # b2 = 0.0
            
            # For a pure resonant filter, more commonly:
            R = np.exp(-np.pi * BW / self.sample_rate)
            omega = 2 * np.pi * F / self.sample_rate
            
            a1 = -2 * R * np.cos(omega)
            a2 = R**2
            
            # Numerator coefficients for a resonant filter, can be simplified to 1 for basic
            # For a gain-compensated peak, g = 1, we can use:
            b0 = (1 - R**2) # Simplified for gain, ensures 0dB peak
            b1 = 0.0
            b2 = 0.0

            # Ensure a0 is 1.0 for biquad standard form (y[n] = b0*x[n] + ... - a1*y[n-1]...)
            a0 = 1.0 
            
            b_coeffs = np.array([b0, b1, b2], dtype=config.FLOATX)
            a_coeffs = np.array([a0, a1, a2], dtype=config.FLOATX)

            self.formant_filters.append(BiquadFilter(b_coeffs, a_coeffs))
            # Reset filter state on update
            self.formant_filters[-1].reset_state()

    def _generate_glottal_source_sample(self, current_f0: config.FLOATX) -> config.FLOATX:
        """
        Generates a single sample of the glottal source signal.
        Uses a simple impulse train / modified sawtooth for voiced, and noise for unvoiced.
        """
        # Use dynamic precision intermediate values
        sample_period = config.FLOATX(1.0) / config.FLOATX(self.sample_rate)

        # Voiced Source (approximated as a pulse train/modified sawtooth)
        # _phase cycles from 0 to 1 at current_f0 rate
        self._phase += current_f0 * sample_period
        self._phase = self._phase % config.FLOATX(1.0)  # Keep phase within 0-1

        # Simple pulse train: impulse at phase 0, then decay.
        if self._phase < (sample_period * current_f0):  # Small window around the "start" of a period
            self._glottal_source_state = config.FLOATX(10.0)  # Stronger impulse (was 1.0)
        else:
            # Exponential decay - slower decay for better sustain
            decay_factor = config.FLOATX(np.exp(- (current_f0 / config.FLOATX(2000.0))))
            self._glottal_source_state *= decay_factor

        voiced_source = self._glottal_source_state

        # Unvoiced Source (white noise) - cast to dynamic precision
        unvoiced_source = config.FLOATX(np.random.uniform(-1.0, 1.0))

        # Mix voiced and unvoiced based on Voicing_Mix parameter
        mixed_source = (voiced_source * self.params['Voicing_Mix'] +
                        unvoiced_source * (config.FLOATX(1.0) - self.params['Voicing_Mix']))

        # Add breathiness (just more noise, before filtering)
        mixed_source += config.FLOATX(np.random.uniform(-1.0, 1.0)) * self.params['Breathiness_Amount'] * config.FLOATX(0.1)

        return config.FLOATX(mixed_source)

    def generate_audio_buffer(self, control_params: Dict[str, config.FLOATX]) -> np.ndarray:
        """
        Generates a buffer of audio samples based on the current control parameters.
        Control params are expected to be from Output.py.
        """
        # Update current synthesis parameters
        for param, value in control_params.items():
            if param in self.params:
                self.params[param] = config.FLOATX(value)
            else:
                logger.warning(f"VocalSynth: Unknown control parameter '{param}'. Ignoring.")

        # Ensure formants are updated if frequencies/bandwidths changed
        # This check should be more robust, comparing new vs old params
        self._initialize_formant_filters()

        audio_buffer = np.zeros(self.buffer_size_samples, dtype=config.FLOATX)

        for i in range(self.buffer_size_samples):
            # Apply vibrato to F0
            # Calculate vibrato using dynamic precision operations; cast trig result to FLOATX
            vib_phase = config.FLOATX(2.0 * np.pi) * self.params['Vibrato_Rate'] * config.FLOATX(self.current_time)
            vib = config.FLOATX(np.sin(vib_phase))
            current_f0 = self.params['F0_hz'] + (self.params['Vibrato_Extent'] * vib)
            current_f0 = np.clip(current_f0, config.FLOATX(20.0), config.FLOATX(1000.0)) # Constrain F0 to reasonable range

            # Generate raw glottal source sample
            source_sample = self._generate_glottal_source_sample(current_f0)

            # Apply Spectral Tilt (simple low-pass/high-pass filter based on parameter)
            # A positive tilt makes it brighter (boost highs), negative makes it darker (boost lows).
            # For simplicity:
            # Actual implementation uses filtering or direct source shaping.
            # A direct way would be to just multiply with some frequency dependent response.
            # For now, it will apply a slight gain to the source sample based on its value and the tilt.
            tilt_factor = config.FLOATX(1.0) + (source_sample * self.params['Spectral_Tilt'] * config.FLOATX(0.1)) # Crude application
            filtered_source = config.FLOATX(source_sample * tilt_factor)

            # Apply Vocal Tract Filter (Formant filters in cascade)
            filtered_output = filtered_source
            for formant_filter in self.formant_filters:
                filtered_output = formant_filter.process_sample(filtered_output)
            
            # Apply gain compensation for formant filter attenuation
            # Since we have 4 filters in cascade, each with gain < 1, compensate
            gain_compensation = config.FLOATX(4.0)  # Boost to compensate for cascade attenuation
            filtered_output = config.FLOATX(filtered_output * gain_compensation)

            # Apply Lip/Mouth Radiation (simple high-pass filter for clarity)
            # This is commonly a first-order differentiator or high-pass filter.
            # For simplicity, we can conceptualize it as adding a high-frequency bias.
            # No explicit filter implemented here to keep code leaner.

            # Overall amplitude scaling
            final_sample = config.FLOATX(filtered_output * self.params['Amplitude'])

            audio_buffer[i] = final_sample
            # Advance time (dynamic precision)
            self.current_time += (config.FLOATX(1.0) / config.FLOATX(self.sample_rate))

        # Ensure output is within -1.0 to 1.0 range
        audio_buffer = np.clip(audio_buffer, config.FLOATX(-1.0), config.FLOATX(1.0))
        return audio_buffer.astype(config.FLOATX) # Ensure dynamic precision output

    def set_parameters(self, params: Dict[str, config.FLOATX]):
        """Directly sets synthesis parameters."""
        for k, v in params.items():
            if k in self.params:
                self.params[k] = config.FLOATX(v)
            else:
                logger.warning(f"VocalSynth: Attempted to set unknown parameter '{k}'.")
        self._initialize_formant_filters() # Re-initialize filters if formants changed

    def get_parameters(self) -> Dict[str, config.FLOATX]:
        """Returns current synthesis parameters."""
        return self.params.copy()

    # --- Backward Compatibility Layer ---
    def synthesize_speech(self, language_vector: np.ndarray) -> np.ndarray:
        """Legacy interface expected by main.py.

        Takes an internal language activation vector (ignored for now) and
        generates an audio buffer using current control parameters.
        """
        try:
            control_params = self.get_parameters()
            buf = self.generate_audio_buffer(control_params)
            # Forward directly to in-process output hub
            try:
                inout.vocalsynth_emit(buf, {'source': 'vocalsynth'})
            except Exception as _:
                # pass  # COMMENTED OUT: stub removed
                pass  # Keep for exception handling, but noted as stub
            return buf
        except Exception as e:
            logger.error(f"VocalSynth.synthesize_speech error: {e}")
            return np.zeros(self.buffer_size_samples, dtype=config.FLOATX)

 