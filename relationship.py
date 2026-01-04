# relationship.py - Social Relationship Module

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class RelationshipModule:
    """
    Manages social relationships, trust, bonds, and social dynamics.
    Provides relationship vectors for social cognition.
    """
    
    def __init__(self, relationship_dim: int = 64):
        self.relationship_dim = relationship_dim
        # Relationship database (entity_id -> relationship_vector)
        self.relationships: Dict[str, np.ndarray] = {}
        # Trust baseline
        self.trust_baseline = 0.5
        # Social bond strength
        self.bond_strength = 0.6
        # Relationship history
        self.interaction_history: List[Dict] = []
        
    def update_relationship(self, entity_id: str, interaction_vector: np.ndarray, outcome: float):
        """
        Update relationship with an entity based on interaction.
        """
        if entity_id not in self.relationships:
            self.relationships[entity_id] = np.full(self.relationship_dim, self.trust_baseline, dtype=np.float32)
            
        # Update relationship vector based on outcome
        learning_rate = 0.05
        adjustment = outcome * interaction_vector
        self.relationships[entity_id] += learning_rate * adjustment
        self.relationships[entity_id] = np.clip(self.relationships[entity_id], -1, 1)
        
        # Record interaction
        self.interaction_history.append({
            'entity': entity_id,
            'interaction': interaction_vector,
            'outcome': outcome,
            'timestamp': np.datetime64('now')
        })
        
    def get_relationship_vector(self, entity_id: str) -> np.ndarray:
        """
        Get relationship vector for an entity.
        """
        return self.relationships.get(entity_id, np.full(self.relationship_dim, self.trust_baseline, dtype=np.float32))
    
    def assess_social_bond(self, entity_id: str) -> float:
        """
        Assess strength of social bond with entity.
        Returns bond strength (0-1).
        """
        if entity_id not in self.relationships:
            return 0.0
            
        relationship = self.relationships[entity_id]
        bond_score = np.mean(np.abs(relationship)) * self.bond_strength
        return float(np.clip(bond_score, 0, 1))
    
    def get_social_context(self, entity_ids: List[str]) -> np.ndarray:
        """
        Get aggregated social context from multiple relationships.
        """
        if not entity_ids:
            return np.zeros(self.relationship_dim, dtype=np.float32)
            
        vectors = [self.get_relationship_vector(eid) for eid in entity_ids]
        return np.mean(vectors, axis=0)