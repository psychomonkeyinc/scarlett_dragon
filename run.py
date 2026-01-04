# run.py - Cognition System Launcher for C GUI

import os
import sys

# Unbuffered stdout/stderr for real-time display
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8')
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1, encoding='utf-8')

# Prevent bytecode generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import logging
import threading
import time
import pickle
import importlib
from typing import Optional, List, Dict
import numpy as np

# Ensure we can import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from input import VideoIn, AudioIn
from health import Health
from mind import Mind
from module_metrics import ModuleMetrics

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("main_log.txt"), logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("run")

# Configuration
CONFIG = {
    'CYCLE_TARGET_SEC': 0.01,
    'HEALTH_CHECK_INTERVAL_SEC': 5.0,
    'SAVE_SLOTS': 3,
    'SAVE_PREFIX': 'save_state_',
}

DEFAULT_MODULES = [
    'nn', 'OptiJustinJ', 'cafve', 'sfe', 'som', 'emotion', 'memory', 'mind',
    'itsagirl', 'goals', 'conscience', 'tom', 'health', 'dream', # 'language',
    'attention', 'output', 'vocalsynth', 'world_map', 'sensastion', 'relationship',
    'physical', 'bionet', 'predict'
]

class RunOrchestrator:
    def __init__(self, modules: Optional[List[str]] = None):
        self.modules = modules or DEFAULT_MODULES
        self._running = False
        self._model_thread = None
        self._command_thread = None
        
        self.health_mod = None
        self.video_in = None
        self.audio_in = None
        self.cycle = 0
        self.loaded_modules: Dict[str, object] = {}
        self.mind = None
        self.metrics_reporter = ModuleMetrics()

    def start(self):
        logger.info("Starting RunOrchestrator...")
        self._running = True
        
        # Start command listener (reads from stdin)
        self._command_thread = threading.Thread(target=self._command_listener, daemon=True)
        self._command_thread.start()

        # DO NOT initialize modules yet - wait for LAUNCH from display
        logger.info("Waiting for LAUNCH signal from display...")

    def _init_real_modules(self):
        try:
            # Initialize sensory inputs
            self.video_in = VideoIn(device_index=0)
            self.audio_in = AudioIn(device_index=None)
            self.video_in.start()
            self.audio_in.start()
            
            # Initialize Health
            self.health_mod = Health(som_map_size=(17, 17), unified_cognitive_state_dim=512, emotional_state_dim=512)

            # Load all declared modules (strict)
            self._load_modules()

            # Initialize core Mind (full cognition)
            self.mind = Mind()
            logger.info("Real modules initialized")
        except Exception as e:
            self._running = False
            logger.error(f"Failed to init real modules: {e}")
            raise

    def _load_modules(self):
        for name in self.modules:
            if name in self.loaded_modules:
                continue
            self.loaded_modules[name] = importlib.import_module(name)
            logger.info(f"Module loaded: {name}")

    def _command_listener(self):
        while self._running:
            line = sys.stdin.readline()
            if not line:
                break
            cmd = line.strip()
            if cmd == 'STOP':
                self._running = False
                break
            elif cmd == 'SNAPSHOT':
                # Create final shutdown snapshot
                if self.mind and hasattr(self.mind, 'data_collector'):
                    logger.info("SNAPSHOT command received. Creating final snapshot...")
                    try:
                        self.mind.data_collector.create_shutdown_snapshot(reason="user_snapshot")
                    except Exception as e:
                        logger.error(f"Failed to create snapshot: {e}")
            elif cmd == 'SLEEP':
                # Trigger sleep state with memory consolidation
                logger.info("SLEEP command received. Initiating sleep state...")
                if self.mind and hasattr(self.mind, 'dream'):
                    try:
                        logger.info("Starting sleep consolidation phase...")
                        result = self.mind.dream.execute_sleep_phase()
                        logger.info(f"Sleep consolidation phase completed. Result: {result}")
                    except Exception as e:
                        logger.error(f"Failed to execute sleep phase: {e}")
                else:
                    logger.warning("Dream module not available for sleep phase")
            elif cmd == 'LAUNCH':
                # Start cognition loop only once UI confirms camera/mic ready
                if self._model_thread is None or not self._model_thread.is_alive():
                    logger.info("LAUNCH received. Initializing modules and starting cognition loop.")
                    # Initialize modules NOW (after LAUNCH)
                    self._init_real_modules()
                    self._model_thread = threading.Thread(target=self._run_cognition_loop, daemon=True)
                    self._model_thread.start()

    def _run_cognition_loop(self):
        logger.info("Cognition loop started")
        last_health_time = time.time()
        
        while self._running:
            start_time = time.time()
            self.cycle += 1
            vis_act = 0.0
            aud_lvl = 0.0
            manual_features = None

            # Build manual sensory feature vector from audio/video inputs (best-effort)
            audio_vec = None
            video_vec = None
            if self.audio_in:
                try:
                    audio_vec = self.audio_in.get_audio_features(as_vector=True)
                except Exception as e:
                    logger.warning(f"Audio feature extraction failed: {e}")
            if self.video_in:
                try:
                    video_vec = self.video_in.get_video_features(as_vector=True)
                except Exception as e:
                    logger.warning(f"Video feature extraction failed: {e}")

            parts = []
            if audio_vec is not None:
                try:
                    parts.append(np.asarray(audio_vec, dtype=np.float32).reshape(-1))
                    aud_lvl = float(np.sqrt(np.mean(np.asarray(audio_vec, dtype=np.float32)**2)))
                except Exception:
                    pass
            if video_vec is not None:
                try:
                    parts.append(np.asarray(video_vec, dtype=np.float32).reshape(-1))
                    vis_act = float(np.mean(np.asarray(video_vec, dtype=np.float32)))
                except Exception:
                    pass
            if parts:
                manual_features = np.concatenate(parts)

            # Run full cognition cycle if Mind initialized; otherwise fallback to telemetry only
            cycle_snapshot = {}
            if self.mind:
                try:
                    cycle_snapshot = self.mind.process_full_cycle(manual_features=manual_features, learning_override=False)
                except Exception as e:
                    logger.error(f"Mind cycle failed: {e}")
                    cycle_snapshot = {"error": str(e)}

            # 2. Update Health (legacy lightweight health ping)
            if time.time() - last_health_time > CONFIG['HEALTH_CHECK_INTERVAL_SEC']:
                if self.health_mod:
                    try:
                        self.health_mod.update(
                            som_bmu_activity=np.zeros((17, 17)),
                            unified_state_vector=np.zeros(512),
                            emotional_state=np.zeros(512)
                        )
                    except Exception as e:
                        logger.error(f"Health update error: {e}")
                last_health_time = time.time()

            # COLLECT DATA: Send snapshot to data collection (every cycle)
            if self.mind and hasattr(self.mind, 'data_collector'):
                try:
                    self.mind.data_collector.capture_snapshot(cycle_snapshot, cycle_snapshot.get("timestamp", time.time()))
                except Exception as e:
                    logger.debug(f"Data collection error: {e}")
            
            # COLLECT MODULE METRICS: Every cycle collect from all modules
            if self.cycle % 10 == 0:  # Print detailed metrics every 10 cycles to avoid spam
                try:
                    metrics = self.metrics_reporter.collect_metrics(
                        mind_instance=self.mind,
                        sfe_features=manual_features,
                        cafve_instance=self.loaded_modules.get('cafve'),
                        som_instance=self.loaded_modules.get('som'),
                        emotion_instance=self.loaded_modules.get('emotion'),
                        memory_instance=self.loaded_modules.get('memory'),
                        output_instance=self.loaded_modules.get('output'),
                        predict_instance=self.loaded_modules.get('predict'),
                        health_instance=self.health_mod,
                        goals_instance=self.loaded_modules.get('goals'),
                        bionet_instance=self.loaded_modules.get('bionet'),
                        vocalsynth_instance=self.loaded_modules.get('vocalsynth')
                    )
                    print(self.metrics_reporter.format_metrics_output(metrics))
                except Exception as e:
                    logger.debug(f"Module metrics error: {e}")

            # 3. Output Metrics for C GUI
            health_score = cycle_snapshot.get("health_score", 0.0)
            cog_norm = cycle_snapshot.get("cognitive_state_norm", 0.0)
            err_flag = 1 if cycle_snapshot.get("processing_complete") is False or "error" in cycle_snapshot else 0
            print(f"[DATA] {int(time.time())}|{self.cycle}|{vis_act:.4f}|0.01|512|{aud_lvl:.4f}|{health_score:.4f}|{cog_norm:.4f}|{err_flag}")
            sys.stdout.flush()

            # 4. Cycle Timing
            elapsed = time.time() - start_time
            if elapsed < CONFIG['CYCLE_TARGET_SEC']:
                time.sleep(CONFIG['CYCLE_TARGET_SEC'] - elapsed)

        logger.info("Cognition loop exiting")

    def stop(self):
        self._running = False
        if self.video_in: self.video_in.stop()
        if self.audio_in: self.audio_in.stop()
        logger.info("Orchestrator stopped")

if __name__ == '__main__':
    orchestrator = RunOrchestrator()
    orchestrator.start()
    
    try:
        while orchestrator._running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.stop()
