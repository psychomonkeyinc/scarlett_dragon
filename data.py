#!/usr/bin/env python3
"""
ACE Maximum Data Collection System
Captures complete consciousness birth data
"""

import json
import time
import numpy as np
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCollection:
    """
    Handles logging of Lillith's consciousness snapshots and internal states.
    Designed for local, append-only logging to JSONL files with proper error handling.
    """
    def __init__(self, 
                 base_dir: Optional[str] = None,  # Auto-detect or use current dir
                 session_id: Optional[str] = None,
                 checkpoint_cycle: int = 1):  # Changed to 1 so data flushes every cycle
        
        self.base_dir = base_dir or os.path.join(os.getcwd(), "data_collection")
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.session_id = session_id or time.strftime("%Y%m%d-%H%M%S")
        self.session_dir = os.path.join(self.base_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.log_path = os.path.join(self.session_dir, "consciousness_stream.jsonl")
        
        self.checkpoint_cycle = checkpoint_cycle
        self.cycle_count = 0
        self.buffer: List[Dict[str, Any]] = []
        
        self._numpy_to_list_lock = threading.Lock()
        self.last_cycle_start_time = time.perf_counter()

        logger.info(f"DataCollection initialized. Logging to: {self.log_path}")

    def _numpy_to_list(self, item: Any) -> Any:
        """
        Recursively converts NumPy arrays within a data structure to Python lists.
        Ensures JSON serializability.
        """
        with self._numpy_to_list_lock:
            if isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, dict):
                return {k: self._numpy_to_list(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [self._numpy_to_list(i) for i in item]
            elif isinstance(item, tuple):
                return [self._numpy_to_list(i) for i in item]  # Convert to list for JSON
            elif isinstance(item, (np.float32, np.float64)):
                return float(item)
            elif isinstance(item, (np.int32, np.int64)):
                return int(item)
            else:
                return item

    def capture_snapshot(self, 
                         snapshot_data: Dict[str, Any], 
                         original_timestamp: float) -> Dict[str, Any]:
        """
        Captures a snapshot of Lillith's consciousness state and adds it to the buffer.
        Automatically flushes the buffer when checkpoint_cycle is reached.
        
        Enhanced to track:
        - CAFVE tokens generated
        - Sensory input stats (video/audio frames processed)
        - Emotional state vectors
        - Output confidence
        - Memory access patterns
        - SOM activation
        """
        try:
            cycle_end_time = time.perf_counter()
            
            serializable_data = self._numpy_to_list(snapshot_data)

            snapshot = {
                "timestamp": time.time(),
                "input_timestamp": original_timestamp,
                "cycle_duration_ms": (cycle_end_time - self.last_cycle_start_time) * 1000,
                "cycle_number": self.cycle_count,
                "data": serializable_data,
                # New enriched fields
                "cafve_tokens": serializable_data.get("cafve_tokens", []),
                "sensory_frame_id": serializable_data.get("sensory_frame_id", 0),
                "emotional_state_stats": {
                    "mean": float(np.mean(serializable_data.get("emotional_state", []))),
                    "std": float(np.std(serializable_data.get("emotional_state", []))),
                    "max": float(np.max(serializable_data.get("emotional_state", [])) if serializable_data.get("emotional_state") else 0)
                } if serializable_data.get("emotional_state") else None,
                "output_confidence": serializable_data.get("output_confidence", 0.0),
                "memory_accessed": serializable_data.get("memory_recalled", False),
                "som_bmu": serializable_data.get("som_bmu_coords", None),
                "prediction_error": serializable_data.get("prediction_error", 0.0),
                "speech_gate_active": serializable_data.get("speech_gate_active", False)
            }
            self.buffer.append(snapshot)
            self.cycle_count += 1
            
            self.last_cycle_start_time = cycle_end_time

            if self.cycle_count >= self.checkpoint_cycle or not os.path.exists(self.log_path):
                self.flush_buffer()
            
            return snapshot
        except Exception as e:
            logger.error(f"Error capturing snapshot: {e}")
            return {}

    def flush_buffer(self):
        """
        Writes buffered snapshots to the JSONL file and clears the buffer.
        """
        if not self.buffer:
            return
        
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                for item in self.buffer:
                    json.dump(item, f, default=str)
                    f.write('\n')
            logger.info(f"Data checkpoint saved ({len(self.buffer)} cycles) to {self.log_path}")
            self.buffer.clear()
            self.cycle_count = 0
        except Exception as e:
            logger.error(f"Error flushing buffer to {self.log_path}: {e}")

    def get_session_directory(self) -> str:
        return self.session_dir

    def get_log_file_path(self) -> str:
        return self.log_path

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "session_id": self.session_id,
            "buffered_cycles": len(self.buffer),
            "total_checkpoints": self.cycle_count,
            "log_path": self.log_path
        }
    def create_shutdown_snapshot(self, reason: str = "shutdown") -> None:
        """
        Creates a final comprehensive snapshot on system shutdown.
        Includes session stats and final state.
        """
        try:
            final_snapshot = {
                "timestamp": time.time(),
                "event": "system_shutdown",
                "reason": reason,
                "total_cycles": self.cycle_count,
                "session_id": self.session_id,
                "session_directory": self.session_dir,
                "log_file": self.log_path
            }
            self.buffer.append(final_snapshot)
            self.flush_buffer()
            logger.info(f"Shutdown snapshot created. Session saved to: {self.session_dir}")
        except Exception as e:
            logger.error(f"Error creating shutdown snapshot: {e}")