"""
Module Metrics Reporter - Collects and outputs data from every module each cycle.
Provides visibility into what each module is actually producing.
"""

import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModuleMetrics:
    """Collects metrics from all cognitive modules each cycle."""
    
    def __init__(self):
        self.cycle_count = 0
        self.stagger_offset = 0  # For staggered sampling
        
    def collect_metrics(self, mind_instance, sfe_features=None, cafve_instance=None, 
                       som_instance=None, emotion_instance=None, memory_instance=None,
                       output_instance=None, predict_instance=None, health_instance=None,
                       goals_instance=None, bionet_instance=None, vocalsynth_instance=None):
        """
        Collect metrics from all modules and return formatted output.
        Returns a dict with all module metrics.
        """
        self.cycle_count += 1
        metrics = {
            'cycle': self.cycle_count,
            'timestamp': datetime.now().isoformat(),
        }
        
        # === SENSORY INPUT ===
        if sfe_features is not None:
            try:
                metrics['sensory'] = {
                    'sfe_mean': float(np.mean(sfe_features)),
                    'sfe_max': float(np.max(sfe_features)),
                    'sfe_std': float(np.std(sfe_features)),
                }
            except Exception as e:
                logger.debug(f"SFE metrics error: {e}")
        
        # === CAFVE TOKENS ===
        if cafve_instance is not None:
            try:
                if hasattr(cafve_instance, 'recent_token_ids'):
                    metrics['cafve'] = {
                        'recent_tokens': len(cafve_instance.recent_token_ids),
                        'vocab_size': len(cafve_instance.vocabulary) if hasattr(cafve_instance, 'vocabulary') else 0,
                    }
                else:
                    metrics['cafve'] = {'recent_tokens': 0, 'vocab_size': 0}
            except Exception as e:
                logger.debug(f"CAFVE metrics error: {e}")
        
        # === SOM (Self-Organizing Map) ===
        if som_instance is not None:
            try:
                activation_map = som_instance.get_activation_map()
                metrics['som'] = {
                    'activation_mean': float(np.mean(activation_map)),
                    'activation_max': float(np.max(activation_map)),
                    'activation_std': float(np.std(activation_map)),
                    'bmu_coords': tuple(som_instance.get_bmu_coords()) if hasattr(som_instance, 'get_bmu_coords') else (0, 0),
                    'fatigue_mean': float(np.mean(som_instance.fatigue)) if hasattr(som_instance, 'fatigue') and som_instance.fatigue is not None else 0.0,
                    'failure_log_count': len(som_instance.failure_log) if hasattr(som_instance, 'failure_log') else 0,
                }
            except Exception as e:
                logger.debug(f"SOM metrics error: {e}")
        
        # === EMOTION ===
        if emotion_instance is not None:
            try:
                if hasattr(emotion_instance, 'current_emotional_state') and emotion_instance.current_emotional_state is not None:
                    state = emotion_instance.current_emotional_state
                    metrics['emotion'] = {
                        'state_mean': float(np.mean(state)),
                        'state_max': float(np.max(state)),
                        'state_min': float(np.min(state)),
                        'arousal': float(emotion_instance.arousal) if hasattr(emotion_instance, 'arousal') else 0.0,
                        'valence': float(emotion_instance.valence) if hasattr(emotion_instance, 'valence') else 0.0,
                    }
                else:
                    metrics['emotion'] = {'state_mean': 0.0, 'state_max': 0.0, 'state_min': 0.0}
            except Exception as e:
                logger.debug(f"Emotion metrics error: {e}")
        
        # === MEMORY ===
        if memory_instance is not None:
            try:
                metrics['memory'] = {
                    'buffer_size': len(memory_instance.sensory_buffer) if hasattr(memory_instance, 'sensory_buffer') else 0,
                    'working_memory_active': len(memory_instance.working_memory) if hasattr(memory_instance, 'working_memory') else 0,
                    'affective_ledger_size': len(memory_instance.affective_ledger) if hasattr(memory_instance, 'affective_ledger') else 0,
                    'associative_weights_norm': float(np.linalg.norm(memory_instance.associative_weights)) if hasattr(memory_instance, 'associative_weights') and memory_instance.associative_weights is not None else 0.0,
                }
            except Exception as e:
                logger.debug(f"Memory metrics error: {e}")
        
        # === BIONET (Biological Systems) ===
        if bionet_instance is not None:
            try:
                metrics['bionet'] = {
                    'energy_gate': float(bionet_instance.energy_gate) if hasattr(bionet_instance, 'energy_gate') else 0.0,
                    'learning_drive': float(bionet_instance.learning_drive) if hasattr(bionet_instance, 'learning_drive') else 0.0,
                    'exploration_bias': float(bionet_instance.exploration_bias) if hasattr(bionet_instance, 'exploration_bias') else 0.0,
                    'stability_gate': float(bionet_instance.stability_gate) if hasattr(bionet_instance, 'stability_gate') else 0.0,
                }
            except Exception as e:
                logger.debug(f"Bionet metrics error: {e}")
        
        # === PREDICTION ===
        if predict_instance is not None:
            try:
                if hasattr(predict_instance, 'prediction_error'):
                    metrics['predict'] = {
                        'prediction_error': float(predict_instance.prediction_error) if predict_instance.prediction_error is not None else 0.0,
                        'last_error_magnitude': float(np.linalg.norm(predict_instance.last_error_vector)) if hasattr(predict_instance, 'last_error_vector') and predict_instance.last_error_vector is not None else 0.0,
                    }
                else:
                    metrics['predict'] = {'prediction_error': 0.0}
            except Exception as e:
                logger.debug(f"Prediction metrics error: {e}")
        
        # === OUTPUT (Vocal Synthesis) ===
        if output_instance is not None:
            try:
                metrics['output'] = {
                    'confidence': float(output_instance.last_confidence) if hasattr(output_instance, 'last_confidence') else 0.0,
                    'f0_hz': float(output_instance.last_f0) if hasattr(output_instance, 'last_f0') else 0.0,
                    'formant_activity': float(np.mean(output_instance.last_formants)) if hasattr(output_instance, 'last_formants') and output_instance.last_formants is not None else 0.0,
                }
            except Exception as e:
                logger.debug(f"Output metrics error: {e}")
        
        # === HEALTH ===
        if health_instance is not None:
            try:
                metrics['health'] = {
                    'overall_score': float(health_instance.overall_health_score()) if hasattr(health_instance, 'overall_health_score') else 0.0,
                    'som_fatigue': float(health_instance.som_fatigue_level) if hasattr(health_instance, 'som_fatigue_level') else 0.0,
                    'failure_count': len(health_instance.failure_log) if hasattr(health_instance, 'failure_log') else 0,
                    'tiredness': float(health_instance.tiredness) if hasattr(health_instance, 'tiredness') else 0.0,
                }
            except Exception as e:
                logger.debug(f"Health metrics error: {e}")
        
        # === GOALS ===
        if goals_instance is not None:
            try:
                satisfaction = goals_instance.get_satisfaction_vector() if hasattr(goals_instance, 'get_satisfaction_vector') else np.zeros(6)
                metrics['goals'] = {
                    'satisfaction_mean': float(np.mean(satisfaction)),
                    'satisfaction_max': float(np.max(satisfaction)),
                    'learning_bias': float(goals_instance.learning_bias) if hasattr(goals_instance, 'learning_bias') else 1.0,
                }
            except Exception as e:
                logger.debug(f"Goals metrics error: {e}")
        
        # === MIND UNIFIED STATE ===
        if mind_instance is not None:
            try:
                if hasattr(mind_instance, 'unified_cognitive_state') and mind_instance.unified_cognitive_state is not None:
                    state = mind_instance.unified_cognitive_state
                    metrics['mind'] = {
                        'unified_state_mean': float(np.mean(state)),
                        'unified_state_max': float(np.max(state)),
                        'unified_state_std': float(np.std(state)),
                        'cycle_duration_ms': float(mind_instance.last_cycle_duration_ms) if hasattr(mind_instance, 'last_cycle_duration_ms') else 0.0,
                    }
                else:
                    metrics['mind'] = {'unified_state_mean': 0.0}
            except Exception as e:
                logger.debug(f"Mind metrics error: {e}")
        
        return metrics
    
    def format_metrics_output(self, metrics):
        """Format metrics dict into readable console output."""
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append(f"CYCLE {metrics['cycle']} | {metrics.get('timestamp', 'N/A')}")
        lines.append(f"{'='*80}")
        
        for module_name, module_data in metrics.items():
            if module_name not in ['cycle', 'timestamp']:
                lines.append(f"\n[{module_name.upper()}]")
                if isinstance(module_data, dict):
                    for key, value in module_data.items():
                        if isinstance(value, float):
                            lines.append(f"  {key:.<30} {value:.4f}")
                        elif isinstance(value, tuple):
                            lines.append(f"  {key:.<30} {value}")
                        else:
                            lines.append(f"  {key:.<30} {value}")
        
        lines.append(f"{'='*80}\n")
        return '\n'.join(lines)


if __name__ == "__main__":
    # Test the metrics reporter
    metrics_reporter = ModuleMetrics()
    
    # Simulate empty metrics
    test_metrics = metrics_reporter.collect_metrics(None, None, None, None, None, None, None, None, None, None, None, None)
    print(metrics_reporter.format_metrics_output(test_metrics))
