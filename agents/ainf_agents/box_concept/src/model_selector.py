#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Selector - Bayesian Model Selection for Object Type Inference

Implements the mathematical framework from Active Inference for treating
obstacle model identity as a discrete latent variable.

From main.tex Section "Model Identity as a Latent Variable":
- Each model m has its own VFE: F_m
- Posterior over model identity: q(m) ∝ p(m) · exp(-F_m)
- Commitment when entropy H[q(m)] falls below threshold

Mathematical Framework:
-----------------------
Generative model: p(o, s, θ, m) = p(o|s,θ_m,m) · p(θ_m|m) · p(m) · p(s)

Variational posterior: q(s, θ, m) = q(s) · q(m) · q(θ_m|m)

Model evidence (VFE): F_m = E_q[ln q(θ_m|m) - ln p(o, s, θ_m|m)]

Posterior over model: q(m) ∝ p(m) · exp(-F_m)

Commitment criteria:
1. Posterior concentration: max_m q(m) > 1 - ε
2. Entropy threshold: H[q(m)] < H_min
3. Hysteresis: N consecutive frames with same winner
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum

from src.belief_core import Belief, BeliefConfig, DEVICE, DTYPE


class ModelState(Enum):
    """State of the model selection process."""
    UNCERTAIN = "uncertain"      # Still evaluating both models
    COMMITTED = "committed"      # Committed to a single model


@dataclass
class ModelSelectorConfig:
    """Configuration for Bayesian Model Selection."""

    # Prior probabilities p(m) for each model type
    # Keys should match model names ('table', 'chair', etc.)
    model_priors: Dict[str, float] = field(default_factory=lambda: {
        'table': 0.5,
        'chair': 0.5
    })

    # Commitment criteria thresholds (increased to delay decision)
    entropy_threshold: float = 0.15         # H[q(m)] below this → commit (was 0.3)
    concentration_threshold: float = 0.92   # max q(m) above this → commit (was 0.85)
    hysteresis_frames: int = 20             # Consecutive frames needed (was 10)

    # VFE scaling (to balance different model complexities)
    vfe_temperature: float = 1.0            # Temperature for softmax

    # Minimum frames before allowing commitment (increased)
    min_frames_before_commit: int = 50      # Was 20

    # Debug output
    debug_interval: int = 20                # Print debug every N frames (was 10)


@dataclass
class ModelHypothesis:
    """Represents one model hypothesis with its belief and statistics."""

    model_type: str                 # 'table', 'chair', etc.
    belief: Belief                  # The actual belief object
    vfe_history: List[float] = field(default_factory=list)
    current_vfe: float = float('inf')

    def update_vfe(self, vfe: float):
        """Update VFE and maintain history."""
        self.current_vfe = vfe
        self.vfe_history.append(vfe)
        # Keep only last 100 values
        if len(self.vfe_history) > 100:
            self.vfe_history = self.vfe_history[-100:]


class MultiModelBelief:
    """
    Maintains multiple model hypotheses for a single detected object.

    This class implements Bayesian Model Selection by:
    1. Running parallel beliefs (one per model type)
    2. Computing q(m) from VFE values
    3. Tracking commitment criteria
    4. Collapsing to single model when appropriate

    Attributes:
        hypotheses: Dict mapping model_type → ModelHypothesis
        q_m: Dict mapping model_type → posterior probability
        state: Current ModelState (UNCERTAIN or COMMITTED)
        committed_model: Model type if committed, None otherwise
    """

    def __init__(self,
                 belief_id: int,
                 hypotheses: Dict[str, ModelHypothesis],
                 config: ModelSelectorConfig = None):
        """
        Initialize multi-model belief.

        Args:
            belief_id: Unique identifier for this belief
            hypotheses: Dict of model_type → ModelHypothesis
            config: Configuration for model selection
        """
        self.id = belief_id
        self.hypotheses = hypotheses
        self.config = config or ModelSelectorConfig()

        # Initialize uniform posterior q(m) based on priors
        self.q_m: Dict[str, float] = {}
        self._initialize_posterior()

        # State tracking
        self.state = ModelState.UNCERTAIN
        self.committed_model: Optional[str] = None

        # Hysteresis tracking
        self.winner_history: List[str] = []
        self.frame_count = 0

        # Statistics
        self.entropy_history: List[float] = []

    def _initialize_posterior(self):
        """Initialize q(m) from prior p(m)."""
        total = sum(self.config.model_priors.get(m, 0.5)
                   for m in self.hypotheses.keys())
        for model_type in self.hypotheses:
            prior = self.config.model_priors.get(model_type, 0.5)
            self.q_m[model_type] = prior / total

    def update_posterior(self):
        """
        Update q(m) based on current VFE values.

        Implements: q(m) ∝ p(m) · exp(-F_m / T)

        Uses log-space computation for numerical stability.
        """
        if self.state == ModelState.COMMITTED:
            # Already committed, no update needed
            return

        # Collect VFE values and priors
        model_types = list(self.hypotheses.keys())
        vfe_values = []
        log_priors = []

        for m in model_types:
            vfe = self.hypotheses[m].current_vfe
            prior = self.config.model_priors.get(m, 0.5)
            vfe_values.append(vfe)
            log_priors.append(np.log(prior + 1e-10))

        vfe_values = np.array(vfe_values)
        log_priors = np.array(log_priors)

        # Compute log q(m) = log p(m) - F_m / T
        T = self.config.vfe_temperature
        log_q = log_priors - vfe_values / T

        # Softmax normalization (in log space for stability)
        log_q_max = np.max(log_q)
        log_sum_exp = log_q_max + np.log(np.sum(np.exp(log_q - log_q_max)))
        log_q_normalized = log_q - log_sum_exp

        # Convert back to probabilities
        q_values = np.exp(log_q_normalized)

        # Update q(m) dict
        for i, m in enumerate(model_types):
            self.q_m[m] = float(q_values[i])

        # Update winner history for hysteresis
        winner = max(self.q_m, key=self.q_m.get)
        self.winner_history.append(winner)
        if len(self.winner_history) > self.config.hysteresis_frames:
            self.winner_history = self.winner_history[-self.config.hysteresis_frames:]

        # Track entropy
        entropy = self.compute_entropy()
        self.entropy_history.append(entropy)

        self.frame_count += 1

    def compute_entropy(self) -> float:
        """
        Compute entropy H[q(m)] = -Σ q(m) log q(m)

        Returns:
            Entropy in nats (natural log)
        """
        entropy = 0.0
        for m, prob in self.q_m.items():
            if prob > 1e-10:
                entropy -= prob * np.log(prob)
        return entropy

    def check_commitment(self) -> Tuple[bool, Optional[str]]:
        """
        Check if commitment criteria are met.

        Returns:
            (should_commit, winning_model_type)
        """
        if self.state == ModelState.COMMITTED:
            return True, self.committed_model

        # Check minimum frames
        if self.frame_count < self.config.min_frames_before_commit:
            return False, None

        # Find current winner
        winner = max(self.q_m, key=self.q_m.get)
        winner_prob = self.q_m[winner]
        entropy = self.compute_entropy()

        # Criterion 1: Posterior concentration
        concentration_met = winner_prob >= self.config.concentration_threshold

        # Criterion 2: Entropy threshold
        entropy_met = entropy < self.config.entropy_threshold

        # Criterion 3: Hysteresis (N consecutive frames with same winner)
        hysteresis_met = (
            len(self.winner_history) >= self.config.hysteresis_frames and
            all(w == winner for w in self.winner_history[-self.config.hysteresis_frames:])
        )

        # Commit if (concentration OR entropy) AND hysteresis
        should_commit = (concentration_met or entropy_met) and hysteresis_met

        return should_commit, winner if should_commit else None

    def commit(self, model_type: str):
        """
        Commit to a specific model type.

        After commitment:
        - State changes to COMMITTED
        - Only the winning model's belief is kept active
        - q(m) becomes degenerate: q(winner) = 1, others = 0

        Args:
            model_type: The model type to commit to
        """
        if model_type not in self.hypotheses:
            raise ValueError(f"Unknown model type: {model_type}")

        self.state = ModelState.COMMITTED
        self.committed_model = model_type

        # Set q(m) to degenerate distribution
        for m in self.q_m:
            self.q_m[m] = 1.0 if m == model_type else 0.0

        # Remove non-committed hypotheses to free memory
        models_to_remove = [m for m in self.hypotheses if m != model_type]
        for m in models_to_remove:
            del self.hypotheses[m]

    def get_active_belief(self) -> Belief:
        """
        Get the currently active belief.

        If committed: returns the committed model's belief
        If uncertain: returns the belief with highest q(m)
        """
        if self.state == ModelState.COMMITTED:
            return self.hypotheses[self.committed_model].belief
        else:
            winner = max(self.q_m, key=self.q_m.get)
            return self.hypotheses[winner].belief

    def get_active_model_type(self) -> str:
        """
        Get the currently active model type.

        If committed: returns the committed model type
        If uncertain: returns the model type with highest q(m)
        """
        if self.state == ModelState.COMMITTED:
            return self.committed_model
        else:
            return max(self.q_m, key=self.q_m.get)

    def get_weighted_position(self) -> Tuple[float, float]:
        """
        Get position weighted by model probabilities.

        For visualization when uncertain, shows weighted average position.
        """
        if self.state == ModelState.COMMITTED:
            belief = self.hypotheses[self.committed_model].belief
            return belief.position

        x_weighted = 0.0
        y_weighted = 0.0
        for m, hyp in self.hypotheses.items():
            prob = self.q_m[m]
            pos = hyp.belief.position
            x_weighted += prob * pos[0]
            y_weighted += prob * pos[1]

        return (x_weighted, y_weighted)

    @property
    def position(self) -> Tuple[float, float]:
        """Position property for compatibility with single-model interface."""
        return self.get_active_belief().position

    @property
    def confidence(self) -> float:
        """Overall confidence considering model uncertainty."""
        active_belief = self.get_active_belief()
        model_confidence = max(self.q_m.values())
        return active_belief.confidence * model_confidence

    def to_dict(self) -> dict:
        """Convert to dictionary for visualization/debugging."""
        active = self.get_active_belief()
        result = active.to_dict()

        # Add model selection info
        result['model_selection'] = {
            'state': self.state.value,
            'committed_model': self.committed_model,
            'q_m': dict(self.q_m),
            'entropy': self.compute_entropy(),
            'frame_count': self.frame_count
        }

        # Add VFE for each hypothesis
        result['model_vfe'] = {
            m: hyp.current_vfe for m, hyp in self.hypotheses.items()
        }

        return result

    def debug_print(self):
        """Print debug information about model selection state."""
        entropy = self.compute_entropy()
        winner = max(self.q_m, key=self.q_m.get)

        print(f"\n{'='*60}")
        print(f"MULTI-MODEL BELIEF {self.id} - Frame {self.frame_count}")
        print(f"{'='*60}")
        print(f"State: {self.state.value}")

        if self.state == ModelState.COMMITTED:
            print(f"Committed to: {self.committed_model}")
        else:
            print(f"Model posteriors q(m):")
            for m, prob in sorted(self.q_m.items(), key=lambda x: -x[1]):
                vfe = self.hypotheses[m].current_vfe
                print(f"  {m:10s}: q={prob:.3f}  VFE={vfe:.4f}")
            print(f"Entropy H[q(m)]: {entropy:.4f} nats")
            print(f"Current winner: {winner} (prob={self.q_m[winner]:.3f})")

            # Hysteresis status
            if self.winner_history:
                recent = self.winner_history[-min(5, len(self.winner_history)):]
                print(f"Recent winners: {recent}")

        print(f"{'='*60}\n")


class ModelSelector:
    """
    Factory and manager for MultiModelBelief objects.

    This class handles:
    - Creating MultiModelBelief from clusters
    - Registering available model types
    - Providing unified interface for the belief manager
    """

    def __init__(self,
                 model_classes: Dict[str, Tuple[Type[Belief], BeliefConfig]],
                 config: ModelSelectorConfig = None,
                 device: torch.device = DEVICE):
        """
        Initialize ModelSelector.

        Args:
            model_classes: Dict mapping model_type → (BeliefClass, config)
                           e.g., {'table': (TableBelief, TableBeliefConfig()),
                                  'chair': (ChairBelief, ChairBeliefConfig())}
            config: Configuration for model selection
            device: Torch device
        """
        self.model_classes = model_classes
        self.config = config or ModelSelectorConfig()
        self.device = device
        self.next_id = 0

    def create_multi_belief(self,
                            cluster: np.ndarray) -> Optional[MultiModelBelief]:
        """
        Create a MultiModelBelief from a cluster.

        Initializes one belief per registered model type.

        Args:
            cluster: [N, 3] points in room frame

        Returns:
            MultiModelBelief or None if no valid hypotheses
        """
        hypotheses = {}

        for model_type, (belief_class, belief_config) in self.model_classes.items():
            # Try to create belief for this model type
            belief = belief_class.from_cluster(
                belief_id=self.next_id,
                cluster=cluster,
                config=belief_config,
                device=self.device
            )

            if belief is not None:
                hypotheses[model_type] = ModelHypothesis(
                    model_type=model_type,
                    belief=belief
                )
                # Debug: Show dimensions for TV
                if model_type == 'tv':
                    print(f"[ModelSelector] TV hypothesis created: "
                          f"size=({belief.width:.2f} x {belief.height:.2f}), "
                          f"z_base={belief.z_base:.2f}, cluster={len(cluster)} pts")

        if not hypotheses:
            return None

        # Create multi-model belief

        # Create multi-model belief
        multi_belief = MultiModelBelief(
            belief_id=self.next_id,
            hypotheses=hypotheses,
            config=self.config
        )

        self.next_id += 1
        return multi_belief

    def update_hypothesis_vfe(self,
                              multi_belief: MultiModelBelief,
                              model_type: str,
                              vfe: float):
        """
        Update VFE for a specific hypothesis.

        Called after optimizing each model's belief.

        Args:
            multi_belief: The MultiModelBelief to update
            model_type: Which model hypothesis
            vfe: The computed VFE value
        """
        if model_type in multi_belief.hypotheses:
            multi_belief.hypotheses[model_type].update_vfe(vfe)

    def process_commitment(self, multi_belief: MultiModelBelief) -> bool:
        """
        Check and process commitment for a multi-belief.

        Args:
            multi_belief: The belief to check

        Returns:
            True if commitment occurred this frame
        """
        should_commit, winner = multi_belief.check_commitment()

        if should_commit and winner is not None:
            multi_belief.commit(winner)
            print(f"[MODEL SELECTOR] Belief {multi_belief.id} committed to: {winner}")
            return True

        return False


# =============================================================================
# Utility functions
# =============================================================================

def compute_model_evidence_ratio(vfe_a: float, vfe_b: float) -> float:
    """
    Compute Bayes factor (evidence ratio) between two models.

    BF = p(o|A) / p(o|B) ≈ exp(F_B - F_A)

    Args:
        vfe_a: VFE of model A
        vfe_b: VFE of model B

    Returns:
        Bayes factor (>1 favors A, <1 favors B)
    """
    # Clamp to avoid overflow
    diff = np.clip(vfe_b - vfe_a, -100, 100)
    return np.exp(diff)


def entropy_to_confidence(entropy: float, n_models: int = 2) -> float:
    """
    Convert entropy to a confidence score in [0, 1].

    Args:
        entropy: H[q(m)] in nats
        n_models: Number of models

    Returns:
        Confidence (1 = certain, 0 = maximum uncertainty)
    """
    max_entropy = np.log(n_models)  # Uniform distribution
    return 1.0 - min(entropy / max_entropy, 1.0)
