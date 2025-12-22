#!/usr/bin/env python3
"""
Probabilistic Epistemic KAT (Knowledge, Actions, and Tests)
Extension of the two-coin model with three types of probabilities:
1. State probabilities (Approach 1): Uncertainty about world state
2. Action probabilities (Approach 2): Stochastic action outcomes
3. Epistemic probabilities (Approach 3): Agent belief uncertainty

Includes HFST export for visualization and manipulation.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Set, Tuple, List, Optional
from itertools import product
from collections import defaultdict

try:
    import hfst
    HFST_AVAILABLE = True
except ImportError:
    HFST_AVAILABLE = False
    print("Warning: HFST not available. Export functionality disabled.")


# ============================================================
# 1. Core Types
# ============================================================

State = Tuple[int, int]  # (coin1, coin2), 0=T, 1=H

ALL_STATES: FrozenSet[State] = frozenset({
    (0, 0), (0, 1), (1, 0), (1, 1)
})


def state_to_bitvector(s: State) -> Tuple[int, int, int, int]:
    """Convert (coin1, coin2) to (c1_H, c1_T, c2_H, c2_T) bit vector."""
    c1_h = 1 if s[0] == 1 else 0
    c1_t = 1 if s[0] == 0 else 0
    c2_h = 1 if s[1] == 1 else 0
    c2_t = 1 if s[1] == 0 else 0
    return (c1_h, c1_t, c2_h, c2_t)


def bitvector_to_state(bv: Tuple[int, int, int, int]) -> State:
    """Convert (c1_H, c1_T, c2_H, c2_T) bit vector to (coin1, coin2)."""
    coin1 = 1 if bv[0] == 1 else 0
    coin2 = 1 if bv[2] == 1 else 0
    return (coin1, coin2)


def bitvector_to_string(bv: Tuple[int, int, int, int]) -> str:
    """Convert bit vector to string: '1001' format."""
    return f"{bv[0]}{bv[1]}{bv[2]}{bv[3]}"


def state_to_string(s: State) -> str:
    """Convert state to bit vector string."""
    return bitvector_to_string(state_to_bitvector(s))


# ============================================================
# 2. Subidentities (Sets of States)
# ============================================================

@dataclass(frozen=True)
class Subidentity:
    """A public-information cell: a set of possible underlying states."""
    states: FrozenSet[State]

    @staticmethod
    def full() -> "Subidentity":
        return Subidentity(ALL_STATES)

    def restrict(self, predicate: Callable[[State], bool]) -> "Subidentity":
        return Subidentity(frozenset(s for s in self.states if predicate(s)))

    def __repr__(self):
        items = ",".join("".join(str(x) for x in st) for st in sorted(self.states))
        return f"{{{items}}}"


# ============================================================
# 3. Probabilistic State Distributions (Approach 1)
# ============================================================

@dataclass
class StateDistribution:
    """Probability distribution over world states."""
    distribution: Dict[State, float]
    
    def __post_init__(self):
        # Normalize
        total = sum(self.distribution.values())
        if abs(total - 1.0) > 1e-6:
            for s in self.distribution:
                self.distribution[s] /= total
    
    @staticmethod
    def uniform() -> "StateDistribution":
        """Uniform distribution over all states."""
        return StateDistribution({s: 0.25 for s in ALL_STATES})
    
    @staticmethod
    def deterministic(state: State) -> "StateDistribution":
        """Deterministic state (100% probability)."""
        return StateDistribution({state: 1.0})
    
    def marginal_coin1(self) -> Dict[int, float]:
        """Marginal distribution over coin1."""
        probs = {0: 0.0, 1: 0.0}
        for state, prob in self.distribution.items():
            probs[state[0]] += prob
        return probs
    
    def marginal_coin2(self) -> Dict[int, float]:
        """Marginal distribution over coin2."""
        probs = {0: 0.0, 1: 0.0}
        for state, prob in self.distribution.items():
            probs[state[1]] += prob
        return probs


# ============================================================
# 4. Epistemic Relations
# ============================================================

@dataclass(frozen=True)
class EpistemicRelation:
    """An agent's equivalence relation on underlying states."""
    name: str
    classes: FrozenSet[FrozenSet[State]]


def partition_by(key_fn: Callable[[State], int]) -> FrozenSet[FrozenSet[State]]:
    buckets: Dict[int, Set[State]] = {}
    for s in ALL_STATES:
        k = key_fn(s)
        buckets.setdefault(k, set()).add(s)
    return frozenset(frozenset(b) for b in buckets.values())


# Amy sees coin1 exactly
Amy_rel = EpistemicRelation(
    "Amy",
    classes=partition_by(lambda st: st[0])
)

# Bob sees coin2 exactly
Bob_rel = EpistemicRelation(
    "Bob",
    classes=partition_by(lambda st: st[1])
)


# ============================================================
# 5. Probabilistic Actions (Approach 2)
# ============================================================

@dataclass(frozen=True)
class ProbabilisticAction:
    """
    Stochastic action semantics.
    rel(s): returns Dict[State, float] - distribution over successor states
    alt[agent]: set of action labels agent may confuse this with
    """
    name: str
    stochastic_rel: Callable[[State], Dict[State, float]]
    alt: Dict[str, Set[str]]
    
    def is_deterministic(self) -> bool:
        """Check if action is deterministic."""
        for s in ALL_STATES:
            successors = self.stochastic_rel(s)
            if len(successors) > 1:
                return False
        return True


# ============================================================
# 6. Action Constructors
# ============================================================

def deterministic_action(
    update_fn: Callable[[State], State],
    precond_fn: Callable[[State], bool] = lambda s: True,
    name: str = "action"
) -> ProbabilisticAction:
    """Create deterministic action (probability 1.0)."""
    def stochastic_rel(s: State) -> Dict[State, float]:
        if precond_fn(s):
            return {update_fn(s): 1.0}
        return {}
    
    return ProbabilisticAction(
        name=name,
        stochastic_rel=stochastic_rel,
        alt={"Amy": {name}, "Bob": {name}}
    )


def identity_action(
    precond_fn: Callable[[State], bool] = lambda s: True,
    name: str = "test"
) -> ProbabilisticAction:
    """Create identity/test action (stays in same state)."""
    def stochastic_rel(s: State) -> Dict[State, float]:
        if precond_fn(s):
            return {s: 1.0}
        return {}
    
    return ProbabilisticAction(
        name=name,
        stochastic_rel=stochastic_rel,
        alt={"Amy": {name}, "Bob": {name}}
    )


def noisy_observation_action(
    precond_fn: Callable[[State], bool],
    accuracy: float,
    name_correct: str,
    name_wrong: str,
    agent: str,
    other_agent: str
) -> Tuple[ProbabilisticAction, ProbabilisticAction]:
    """
    Create pair of noisy observation actions (correct and wrong outcomes).
    
    The correct action applies when precond is true (correct observation).
    The wrong action applies when precond is true (wrong observation).
    
    Returns:
        (correct_action, wrong_action)
    """
    def correct_rel(s: State) -> Dict[State, float]:
        # Correct observation occurs with probability 'accuracy' when precond is true
        if precond_fn(s):
            return {s: accuracy}
        # When precond is false, this is a wrong observation (error case)
        # This happens with probability (1-accuracy)
        return {s: 1.0 - accuracy}
    
    def wrong_rel(s: State) -> Dict[State, float]:
        # Wrong observation occurs with probability (1-accuracy) when precond is true
        if precond_fn(s):
            return {s: 1.0 - accuracy}
        # When precond is false, this becomes the "correct" observation for false case
        # This happens with probability 'accuracy'
        return {s: accuracy}
    
    correct_action = ProbabilisticAction(
        name=name_correct,
        stochastic_rel=correct_rel,
        alt={agent: {name_correct}, other_agent: {name_correct, name_wrong}}
    )
    
    wrong_action = ProbabilisticAction(
        name=name_wrong,
        stochastic_rel=wrong_rel,
        alt={agent: {name_wrong}, other_agent: {name_correct, name_wrong}}
    )
    
    return correct_action, wrong_action


# ============================================================
# 7. Epistemic Belief Distributions (Approach 3)
# ============================================================

@dataclass
class EpistemicBeliefState:
    """
    Agent's probabilistic beliefs over equivalence classes.
    Maps each equivalence class to the probability the true state is in it.
    """
    agent: str
    belief_distribution: Dict[FrozenSet[State], float]
    
    def __post_init__(self):
        # Normalize
        total = sum(self.belief_distribution.values())
        if abs(total - 1.0) > 1e-6:
            for ec in self.belief_distribution:
                self.belief_distribution[ec] /= total
    
    def entropy(self) -> float:
        """Compute Shannon entropy of belief distribution."""
        entropy = 0.0
        for prob in self.belief_distribution.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy
    
    def most_likely_class(self) -> Tuple[FrozenSet[State], float]:
        """Return most likely equivalence class and its probability."""
        return max(self.belief_distribution.items(), key=lambda x: x[1])


def bayesian_update(
    prior: EpistemicBeliefState,
    observation_likelihood: Dict[FrozenSet[State], float]
) -> EpistemicBeliefState:
    """
    Perform Bayesian update of beliefs given observation likelihoods.
    
    Args:
        prior: Prior belief distribution
        observation_likelihood: P(observation | equivalence_class)
    """
    posterior = {}
    
    # Bayes rule: P(class | obs) ∝ P(obs | class) * P(class)
    for equiv_class, prior_prob in prior.belief_distribution.items():
        likelihood = observation_likelihood.get(equiv_class, 0.0)
        posterior[equiv_class] = likelihood * prior_prob
    
    return EpistemicBeliefState(
        agent=prior.agent,
        belief_distribution=posterior
    )


# ============================================================
# 8. Concrete Actions for Two-Coin Model
# ============================================================

# Preconditions
def pre_coin1_H(s: State) -> bool: return s[0] == 1
def pre_coin1_T(s: State) -> bool: return s[0] == 0
def pre_coin2_H(s: State) -> bool: return s[1] == 1
def pre_coin2_T(s: State) -> bool: return s[1] == 0


# Deterministic announcements (test actions)
announce_coin1_H = identity_action(pre_coin1_H, "announce_coin1_H")
announce_coin1_T = identity_action(pre_coin1_T, "announce_coin1_T")
announce_coin2_H = identity_action(pre_coin2_H, "announce_coin2_H")
announce_coin2_T = identity_action(pre_coin2_T, "announce_coin2_T")


# Perfect observations (deterministic)
peek_Amy_coin1_H_perfect = identity_action(pre_coin1_H, "peek_Amy_coin1_H")
peek_Amy_coin1_T_perfect = identity_action(pre_coin1_T, "peek_Amy_coin1_T")
peek_Amy_coin2_H_perfect = identity_action(pre_coin2_H, "peek_Amy_coin2_H")
peek_Amy_coin2_T_perfect = identity_action(pre_coin2_T, "peek_Amy_coin2_T")


# Noisy observations (stochastic, 90% accurate)
def create_noisy_peeks(accuracy: float = 0.9):
    """Create noisy observation actions for both agents and both coins."""
    
    # Amy peeks coin1
    peek_Amy_coin1_H, peek_Amy_coin1_T = noisy_observation_action(
        pre_coin1_H, accuracy,
        "peek_Amy_coin1_H_noisy", "peek_Amy_coin1_T_noisy",
        "Amy", "Bob"
    )
    
    # Amy peeks coin2
    peek_Amy_coin2_H, peek_Amy_coin2_T = noisy_observation_action(
        pre_coin2_H, accuracy,
        "peek_Amy_coin2_H_noisy", "peek_Amy_coin2_T_noisy",
        "Amy", "Bob"
    )
    
    # Bob peeks coin1
    peek_Bob_coin1_H, peek_Bob_coin1_T = noisy_observation_action(
        pre_coin1_H, accuracy,
        "peek_Bob_coin1_H_noisy", "peek_Bob_coin1_T_noisy",
        "Bob", "Amy"
    )
    
    # Bob peeks coin2
    peek_Bob_coin2_H, peek_Bob_coin2_T = noisy_observation_action(
        pre_coin2_H, accuracy,
        "peek_Bob_coin2_H_noisy", "peek_Bob_coin2_T_noisy",
        "Bob", "Amy"
    )
    
    return {
        "peek_Amy_coin1_H_noisy": peek_Amy_coin1_H,
        "peek_Amy_coin1_T_noisy": peek_Amy_coin1_T,
        "peek_Amy_coin2_H_noisy": peek_Amy_coin2_H,
        "peek_Amy_coin2_T_noisy": peek_Amy_coin2_T,
        "peek_Bob_coin1_H_noisy": peek_Bob_coin1_H,
        "peek_Bob_coin1_T_noisy": peek_Bob_coin1_T,
        "peek_Bob_coin2_H_noisy": peek_Bob_coin2_H,
        "peek_Bob_coin2_T_noisy": peek_Bob_coin2_T,
    }


# World-changing action
def upd_turn_1_HT(s: State) -> State:
    return (0, s[1])  # Flip coin1 from H to T


turn_1_HT = deterministic_action(
    upd_turn_1_HT,
    pre_coin1_H,
    "turn_1_HT"
)


# ============================================================
# 9. Unified Probabilistic Model
# ============================================================

@dataclass
class ProbabilisticTwoCoinModel:
    """
    Complete two-coin model with three types of probabilities:
    1. State distributions (Approach 1)
    2. Stochastic actions (Approach 2)
    3. Epistemic beliefs (Approach 3)
    """
    agents: FrozenSet[str]
    epistemic: Dict[str, EpistemicRelation]
    actions: Dict[str, ProbabilisticAction]
    
    # Approach 1: State probability
    initial_state_dist: StateDistribution = field(
        default_factory=StateDistribution.uniform
    )
    
    # Approach 3: Agent beliefs
    agent_beliefs: Dict[str, EpistemicBeliefState] = field(
        default_factory=dict
    )
    
    def get_action_successors(
        self, 
        state: State, 
        action_name: str
    ) -> Dict[State, float]:
        """Get probabilistic successors of state under action."""
        action = self.actions[action_name]
        return action.stochastic_rel(state)
    
    def compute_trajectory_probability(
        self,
        action_sequence: List[str],
        initial_state: State
    ) -> Dict[State, float]:
        """
        Compute distribution over final states given action sequence.
        Combines Approach 1 (state dist) and Approach 2 (action probs).
        """
        # Start with deterministic initial state
        current_dist = {initial_state: 1.0}
        
        for action_name in action_sequence:
            next_dist = defaultdict(float)
            
            for state, state_prob in current_dist.items():
                successors = self.get_action_successors(state, action_name)
                
                for next_state, action_prob in successors.items():
                    next_dist[next_state] += state_prob * action_prob
            
            current_dist = dict(next_dist)
        
        return current_dist


# ============================================================
# 10. HFST Export Functions
# ============================================================

def prob_to_weight(p: float) -> float:
    """Convert probability to HFST weight (tropical semiring)."""
    return -math.log(p) if p > 0 else float('inf')


def weight_to_prob(w) -> float:
    """
    Convert HFST weight back to probability.
    
    Args:
        w: Weight from HFST, can be float or list of floats
    
    Returns:
        Probability value
    """
    # Handle list weights (HFST sometimes returns weights as lists)
    if isinstance(w, (list, tuple)):
        if len(w) == 0:
            return 0.0
        # Use first weight (for single-weight transducers)
        w = w[0]
    
    # Convert weight to probability
    if w == float('inf'):
        return 0.0
    try:
        return math.exp(-w)
    except (TypeError, OverflowError):
        return 0.0


def extract_paths_safe(transducer: 'hfst.HfstTransducer', max_cycles: int = 1, max_paths: int = 100):
    """
    Safely extract paths from a transducer that may have cycles.
    
    Args:
        transducer: The HFST transducer
        max_cycles: Maximum number of times to traverse cycles
        max_paths: Maximum number of paths to extract
    
    Returns:
        Dictionary mapping path strings to weights (normalized to floats)
    """
    try:
        # Try to extract paths with cycle limit
        paths = transducer.extract_paths(output='dict', max_cycles=max_cycles)
        
        # Normalize weights to floats
        # HFST can return: [(path_string, weight_value)]
        normalized_paths = {}
        for path, weight in paths.items():
            # Handle different weight formats
            # Format 1: [(path_str, weight_float)]
            if isinstance(weight, list) and len(weight) > 0:
                if isinstance(weight[0], tuple) and len(weight[0]) >= 2:
                    # Extract weight from (path, weight) tuple
                    weight = weight[0][1]
            
            # Format 2: Nested lists/tuples with numeric at the end
            while isinstance(weight, (list, tuple)) and len(weight) > 0:
                # If it's a tuple with 2 elements where second is numeric, use that
                if isinstance(weight, tuple) and len(weight) == 2:
                    try:
                        float(weight[1])
                        weight = weight[1]
                        break
                    except (TypeError, ValueError):
                        pass
                # Otherwise unwrap first element
                weight = weight[0]
            
            # Convert to float
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = float('inf')
            
            normalized_paths[path] = weight
        
        # Limit number of paths if too many
        if len(normalized_paths) > max_paths:
            # Keep only the lowest-weight (highest-probability) paths
            sorted_paths = sorted(normalized_paths.items(), key=lambda x: x[1])[:max_paths]
            normalized_paths = dict(sorted_paths)
        
        return normalized_paths
    except Exception as e:
        print(f"Warning: Could not extract paths from cyclic transducer: {e}")
        return {}



if HFST_AVAILABLE:
    
    def export_state_distribution_to_hfst(
        state_dist: StateDistribution,
        add_self_loops: bool = True
    ) -> hfst.HfstTransducer:
        """
        Approach 1: Export probability distribution over states.
        
        Structure:
        - Initial state branches to one state per possible world
        - Each branch weighted by P(world_state)
        - Each world state has self-loop (identity) if add_self_loops=True
        
        Args:
            state_dist: The state distribution to export
            add_self_loops: If True, add self-loops (creates cycles).
                           If False, create acyclic FST (better for path extraction).
        """
        tr = hfst.HfstBasicTransducer()
        tr.add_state(0)  # Initial state
        
        for idx, (state, prob) in enumerate(sorted(state_dist.distribution.items()), 1):
            tr.add_state(idx)
            
            # Transition to this world with its probability
            state_symbol = state_to_string(state)
            weight = prob_to_weight(prob)
            tr.add_transition(0, idx, state_symbol, state_symbol, weight)
            
            if add_self_loops:
                # Self-loop to maintain world state (creates cycles)
                tr.add_transition(idx, idx, state_symbol, state_symbol, 0.0)
            
            tr.set_final_weight(idx, 0.0)
        
        return hfst.HfstTransducer(tr)
    
    
    def export_action_to_hfst(
        action: ProbabilisticAction,
        encode_alternatives: bool = False
    ) -> hfst.HfstTransducer:
        """
        Approach 2: Export probabilistic action as HFST transducer.
        
        Structure:
        - Input: world state symbol
        - Output: world state + action label + next world state
        - Weight: action probability
        """
        tr = hfst.HfstBasicTransducer()
        
        # Create state for each possible world configuration
        state_map = {}
        for idx, state in enumerate(sorted(ALL_STATES)):
            tr.add_state(idx)
            state_map[state] = idx
            tr.set_final_weight(idx, 0.0)
        
        # Add transitions for each state
        for state, state_idx in state_map.items():
            in_symbol = state_to_string(state)
            
            # Get stochastic successors
            successors = action.stochastic_rel(state)
            
            for next_state, prob in successors.items():
                next_idx = state_map[next_state]
                
                # Output symbol: state + action + next_state
                out_symbol = f"{in_symbol}{action.name}{state_to_string(next_state)}"
                weight = prob_to_weight(prob)
                
                tr.add_transition(
                    state_idx, next_idx,
                    in_symbol, out_symbol,
                    weight
                )
        
        return hfst.HfstTransducer(tr)
    
    
    def export_epistemic_beliefs_to_hfst(
        belief_state: EpistemicBeliefState,
        epistemic_rel: EpistemicRelation
    ) -> hfst.HfstTransducer:
        """
        Approach 3: Export agent's probabilistic beliefs.
        
        Structure:
        - States represent belief distributions
        - Transitions encode belief updates
        - Weights represent P(belief | prior, observation)
        """
        tr = hfst.HfstBasicTransducer()
        
        # Create state for each equivalence class with its belief
        class_to_idx = {}
        for idx, (equiv_class, confidence) in enumerate(
            sorted(belief_state.belief_distribution.items(),
                   key=lambda x: tuple(sorted(x[0])))
        ):
            tr.add_state(idx)
            class_to_idx[equiv_class] = idx
            
            # Encode belief as symbol
            class_states = "_".join(state_to_string(s) for s in sorted(equiv_class))
            belief_symbol = f"{belief_state.agent}_bel:{class_states}:conf{int(confidence*100)}"
            
            # Self-loop with belief weight
            weight = prob_to_weight(confidence)
            tr.add_transition(idx, idx, belief_symbol, belief_symbol, weight)
            tr.set_final_weight(idx, 0.0)
        
        return hfst.HfstTransducer(tr)
    
    
    def export_complete_model_to_hfst(
        model: ProbabilisticTwoCoinModel,
        action_sequence: List[str],
        include_beliefs: bool = True
    ) -> hfst.HfstTransducer:
        """
        Export complete model combining all three approaches.
        
        Args:
            model: The probabilistic model
            action_sequence: Sequence of actions to compose
            include_beliefs: Whether to include epistemic beliefs (Approach 3)
        """
        # Layer 1: Initial state distribution
        T_states = export_state_distribution_to_hfst(model.initial_state_dist)
        
        # Layer 2: Compose action sequence
        T_complete = T_states.copy()
        
        for action_name in action_sequence:
            if action_name not in model.actions:
                raise ValueError(f"Unknown action: {action_name}")
            
            action = model.actions[action_name]
            T_action = export_action_to_hfst(action)
            
            T_complete.compose(T_action)
            T_complete.minimize()
        
        # Layer 3: Add epistemic beliefs if requested
        if include_beliefs and model.agent_beliefs:
            for agent_name, belief_state in model.agent_beliefs.items():
                epistemic_rel = model.epistemic[agent_name]
                T_beliefs = export_epistemic_beliefs_to_hfst(belief_state, epistemic_rel)
                
                T_complete.compose(T_beliefs)
                T_complete.minimize()
        
        return T_complete


# ============================================================
# 11. Model Builder
# ============================================================

def build_complete_model(
    include_noisy_actions: bool = True,
    noise_accuracy: float = 0.9
) -> ProbabilisticTwoCoinModel:
    """
    Build complete two-coin model with all action types.
    
    Args:
        include_noisy_actions: Whether to include noisy observations
        noise_accuracy: Accuracy of noisy observations (0.9 = 90% accurate)
    """
    actions = {
        "announce_coin1_H": announce_coin1_H,
        "announce_coin1_T": announce_coin1_T,
        "announce_coin2_H": announce_coin2_H,
        "announce_coin2_T": announce_coin2_T,
        "peek_Amy_coin1_H": peek_Amy_coin1_H_perfect,
        "peek_Amy_coin1_T": peek_Amy_coin1_T_perfect,
        "peek_Amy_coin2_H": peek_Amy_coin2_H_perfect,
        "peek_Amy_coin2_T": peek_Amy_coin2_T_perfect,
        "turn_1_HT": turn_1_HT,
    }
    
    if include_noisy_actions:
        noisy_actions = create_noisy_peeks(noise_accuracy)
        actions.update(noisy_actions)
    
    # Initialize with uniform distribution
    initial_dist = StateDistribution.uniform()
    
    # Initialize agent beliefs (uniform over equivalence classes)
    agent_beliefs = {
        "Amy": EpistemicBeliefState(
            agent="Amy",
            belief_distribution={
                frozenset([(1, 0), (1, 1)]): 0.5,  # coin1=H
                frozenset([(0, 0), (0, 1)]): 0.5,  # coin1=T
            }
        ),
        "Bob": EpistemicBeliefState(
            agent="Bob",
            belief_distribution={
                frozenset([(0, 1), (1, 1)]): 0.5,  # coin2=H
                frozenset([(0, 0), (1, 0)]): 0.5,  # coin2=T
            }
        ),
    }
    
    return ProbabilisticTwoCoinModel(
        agents=frozenset({"Amy", "Bob"}),
        epistemic={
            "Amy": Amy_rel,
            "Bob": Bob_rel,
        },
        actions=actions,
        initial_state_dist=initial_dist,
        agent_beliefs=agent_beliefs
    )


if __name__ == "__main__":
    print("Probabilistic Epistemic KAT Model")
    print("=" * 50)
    
    # Build model
    model = build_complete_model(include_noisy_actions=True, noise_accuracy=0.9)
    
    print(f"\nModel has {len(model.actions)} actions:")
    for name in sorted(model.actions.keys()):
        action = model.actions[name]
        det = "deterministic" if action.is_deterministic() else "stochastic"
        print(f"  - {name} ({det})")
    
    print(f"\nInitial state distribution:")
    for state, prob in sorted(model.initial_state_dist.distribution.items()):
        print(f"  {state}: {prob:.3f}")
    
    # Test trajectory computation
    print(f"\nExample trajectory:")
    initial_state = (1, 0)  # coin1=H, coin2=T
    actions = ["peek_Amy_coin1_H_noisy"]
    
    final_dist = model.compute_trajectory_probability(actions, initial_state)
    print(f"Starting from {initial_state}, after {actions}:")
    for state, prob in sorted(final_dist.items()):
        print(f"  {state}: {prob:.3f}")
    
    if HFST_AVAILABLE:
        print("\nHFST export available - ready to generate transducers!")
    else:
        print("\nHFST not available - install hfst-python for export functionality")