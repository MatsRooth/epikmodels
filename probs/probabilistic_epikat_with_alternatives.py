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

alt_dict = {
    "announce_coin1_H": {"Amy": {"announce_coin1_H"}, "Bob": {"announce_coin1_H"}},
    "announce_coin1_T": {"Amy": {"announce_coin1_T"}, "Bob": {"announce_coin1_T"}},
    "announce_coin2_H": {"Amy": {"announce_coin2_H"}, "Bob": {"announce_coin2_H"}},
    "announce_coin2_T": {"Amy": {"announce_coin2_T"}, "Bob": {"announce_coin2_T"}},
    "announce_coin1_H": {"Amy": {"announce_coin1_H"}, "Bob": {"announce_coin1_H"}},
    "announce_coin1_T": {"Amy": {"announce_coin1_T"}, "Bob": {"announce_coin1_T"}},
    "announce_coin2_H": {"Amy": {"announce_coin2_H"}, "Bob": {"announce_coin2_H"}},
    "announce_coin2_T": {"Amy": {"announce_coin2_T"}, "Bob": {"announce_coin2_T"}},
    "peek_Amy_coin1_H": {"Amy": {"peek_Amy_coin1_H"}, "Bob": {"peek_Amy_coin1_H", "peek_Amy_coin1_T"}},
    "peek_Amy_coin1_T": {"Amy": {"peek_Amy_coin1_T"}, "Bob": {"peek_Amy_coin1_H", "peek_Amy_coin1_T"}},
    "peek_Amy_coin2_H": {"Amy": {"peek_Amy_coin2_H"}, "Bob": {"peek_Amy_coin2_H", "peek_Amy_coin2_T"}},
    "peek_Amy_coin2_T": {"Amy": {"peek_Amy_coin2_T"}, "Bob": {"peek_Amy_coin2_H", "peek_Amy_coin2_T"}},
    "peek_Bob_coin1_H": {"Bob": {"peek_Bob_coin1_H"}, "Amy": {"peek_Bob_coin1_H", "peek_Bob_coin1_T"}},
    "peek_Bob_coin1_T": {"Bob": {"peek_Bob_coin1_T"}, "Amy": {"peek_Bob_coin1_H", "peek_Bob_coin1_T"}},
    "peek_Bob_coin2_H": {"Bob": {"peek_Bob_coin2_H"}, "Amy": {"peek_Bob_coin2_H", "peek_Bob_coin2_T"}},
    "peek_Bob_coin2_T": {"Bob": {"peek_Bob_coin2_T"}, "Amy": {"peek_Bob_coin2_H", "peek_Bob_coin2_T"}},
    "amy_flip_coin_1_HT": {"Amy": {"amy_flip_coin_1_HT", "amy_flip_coin_1_TH"}, "Bob": {"amy_flip_coin_1_HT", "amy_flip_coin_1_TH"}},
    "bob_flip_coin_1_HT": {"Bob": {"bob_flip_coin_1_HT", "bob_flip_coin_1_TH"}, "Amy": {"bob_flip_coin_1_HT", "bob_flip_coin_1_TH"}},
    "amy_flip_coin_1_TH": {"Amy": {"amy_flip_coin_1_TH", "amy_flip_coin_1_HT"}, "Bob": {"amy_flip_coin_1_HT", "amy_flip_coin_1_TH"}},
    "bob_flip_coin_1_TH": {"Bob": {"bob_flip_coin_1_TH", "bob_flip_coin_1_HT"}, "Amy": {"bob_flip_coin_1_HT", "bob_flip_coin_1_TH"}},
}

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
    action_name: str
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
    action_name: str = "action_name",
    alt_dict: Dict[str, Set[str]] = {'agent': {'action_name1', 'action_name2'}}
) -> ProbabilisticAction:
    """Create deterministic action (probability 1.0)."""
    def stochastic_rel(s: State) -> Dict[State, float]:
        if precond_fn(s):
            return {update_fn(s): 1.0}
        return {}
    
    return ProbabilisticAction(
        action_name=action_name,
        stochastic_rel=stochastic_rel,
        alt=alt_dict
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
        action_name=name_correct,
        stochastic_rel=correct_rel,
        alt={agent: {name_correct}, other_agent: {name_correct, name_wrong}}
    )
    
    wrong_action = ProbabilisticAction(
        action_name=name_wrong,
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
def post_peek(s: State) -> State: return (s[0], s[1])

# Deterministic announcements (test actions)
def create_deterministic_announcements():
    # alt_dict = {
    #     "announce_coin1_H": {"Amy": {"announce_coin1_H"}, "Bob": {"announce_coin1_H"}},
    #     "announce_coin1_T": {"Amy": {"announce_coin1_T"}, "Bob": {"announce_coin1_T"}},
    #     "announce_coin2_H": {"Amy": {"announce_coin2_H"}, "Bob": {"announce_coin2_H"}},
    #     "announce_coin2_T": {"Amy": {"announce_coin2_T"}, "Bob": {"announce_coin2_T"}},
    # }
    announce_coin1_H = deterministic_action(post_peek, pre_coin1_H, "announce_coin1_H", alt_dict["announce_coin1_H"])
    announce_coin1_T = deterministic_action(post_peek, pre_coin1_T, "announce_coin1_T", alt_dict["announce_coin1_T"])
    announce_coin2_H = deterministic_action(post_peek, pre_coin2_H, "announce_coin2_H", alt_dict["announce_coin2_H"])
    announce_coin2_T = deterministic_action(post_peek, pre_coin2_T, "announce_coin2_T", alt_dict["announce_coin2_T"])
    return {
        "announce_coin1_H": announce_coin1_H,
        "announce_coin1_T": announce_coin1_T,
        "announce_coin2_H": announce_coin2_H,
        "announce_coin2_T": announce_coin2_T,
    }

# Deterministic peeks
# TODO: what if when Amy peeks Bob already knows the outcome?
# TODO: what if when Bob peeks Amy already knows the outcome?
def create_deterministic_peeks():
    # alt_dict = {
    #     "announce_coin1_H": {"Amy": {"announce_coin1_H"}, "Bob": {"announce_coin1_H"}},
    #     "announce_coin1_T": {"Amy": {"announce_coin1_T"}, "Bob": {"announce_coin1_T"}},
    #     "announce_coin2_H": {"Amy": {"announce_coin2_H"}, "Bob": {"announce_coin2_H"}},
    #     "announce_coin2_T": {"Amy": {"announce_coin2_T"}, "Bob": {"announce_coin2_T"}},
    #     "peek_Amy_coin1_H": {"Amy": {"peek_Amy_coin1_H"}, "Bob": {"peek_Amy_coin1_H", "peek_Amy_coin1_T"}},
    #     "peek_Amy_coin1_T": {"Amy": {"peek_Amy_coin1_T"}, "Bob": {"peek_Amy_coin1_H", "peek_Amy_coin1_T"}},
    #     "peek_Amy_coin2_H": {"Amy": {"peek_Amy_coin2_H"}, "Bob": {"peek_Amy_coin2_H", "peek_Amy_coin2_T"}},
    #     "peek_Amy_coin2_T": {"Amy": {"peek_Amy_coin2_T"}, "Bob": {"peek_Amy_coin2_H", "peek_Amy_coin2_T"}},
    #     "peek_Bob_coin1_H": {"Bob": {"peek_Bob_coin1_H"}, "Amy": {"peek_Bob_coin1_H", "peek_Bob_coin1_T"}},
    #     "peek_Bob_coin1_T": {"Bob": {"peek_Bob_coin1_T"}, "Amy": {"peek_Bob_coin1_H", "peek_Bob_coin1_T"}},
    #     "peek_Bob_coin2_H": {"Bob": {"peek_Bob_coin2_H"}, "Amy": {"peek_Bob_coin2_H", "peek_Bob_coin2_T"}},
    #     "peek_Bob_coin2_T": {"Bob": {"peek_Bob_coin2_T"}, "Amy": {"peek_Bob_coin2_H", "peek_Bob_coin2_T"}},
    # }

    # Perfect observations (deterministic)
    peek_Amy_coin1_H = deterministic_action(
        post_peek,
        pre_coin1_H,
        "peek_Amy_coin1_H",
        alt_dict["peek_Amy_coin1_H"]
    )
    peek_Amy_coin1_T = deterministic_action(
        post_peek,
        pre_coin1_T,
        "peek_Amy_coin1_T",
        alt_dict["peek_Amy_coin1_T"]
    )
    peek_Amy_coin2_H = deterministic_action(
        post_peek,
        pre_coin2_H,
        "peek_Amy_coin2_H",
        alt_dict["peek_Amy_coin2_H"]
    )
    peek_Amy_coin2_T = deterministic_action(
        post_peek,
        pre_coin2_T,
        "peek_Amy_coin2_T",
        alt_dict["peek_Amy_coin2_T"]
    )
    peek_Bob_coin1_H = deterministic_action(
        post_peek,
        pre_coin1_H,
        "peek_Bob_coin1_H",
        alt_dict["peek_Bob_coin1_H"]
    )
    peek_Bob_coin1_T = deterministic_action(
        post_peek,
        pre_coin1_T,
        "peek_Bob_coin1_T",
        alt_dict["peek_Bob_coin1_T"]
    )
    peek_Bob_coin2_H = deterministic_action(
        post_peek,
        pre_coin2_H,
        "peek_Bob_coin2_H",
        alt_dict["peek_Bob_coin2_H"]
    )
    peek_Bob_coin2_T = deterministic_action(
        post_peek,
        pre_coin2_T,
        "peek_Bob_coin2_T",
        alt_dict["peek_Bob_coin2_T"]
    )
    return {
        "peek_Amy_coin1_H": peek_Amy_coin1_H,
        "peek_Amy_coin1_T": peek_Amy_coin1_T,
        "peek_Amy_coin2_H": peek_Amy_coin2_H,
        "peek_Amy_coin2_T": peek_Amy_coin2_T,
        "peek_Bob_coin1_H": peek_Bob_coin1_H,
        "peek_Bob_coin1_T": peek_Bob_coin1_T,
        "peek_Bob_coin2_H": peek_Bob_coin2_H,
        "peek_Bob_coin2_T": peek_Bob_coin2_T,
    }

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
def flip_coin_1_HT(s: State) -> State:
    return (0, s[1])  # Flip coin1 from H to T

def flip_coin_1_TH(s: State) -> State:
    return (1, s[1])  # Flip coin1 from T to H

def create_deterministic_flip_coin_1():
    # flip_alt_dict = {
        # "amy_flip_coin_1_HT": {"Amy": {"amy_flip_coin_1_HT", "amy_flip_coin_1_TH"}, "Bob": {"amy_flip_coin_1_HT", "amy_flip_coin_1_TH"}},
        # "bob_flip_coin_1_HT": {"Bob": {"bob_flip_coin_1_HT", "bob_flip_coin_1_TH"}, "Amy": {"bob_flip_coin_1_HT", "bob_flip_coin_1_TH"}},
        # "amy_flip_coin_1_TH": {"Amy": {"amy_flip_coin_1_TH", "amy_flip_coin_1_HT"}, "Bob": {"amy_flip_coin_1_HT", "amy_flip_coin_1_TH"}},
        # "bob_flip_coin_1_TH": {"Bob": {"bob_flip_coin_1_TH", "bob_flip_coin_1_HT"}, "Amy": {"bob_flip_coin_1_HT", "bob_flip_coin_1_TH"}},
    # }
    amy_flip_coin_1_HT = deterministic_action(
        flip_coin_1_HT,
        pre_coin1_H,
        "amy_flip_coin_1_HT",
        alt_dict["amy_flip_coin_1_HT"]
    )
    amy_flip_coin_1_TH = deterministic_action(
        flip_coin_1_TH,
        pre_coin1_T,
        "amy_flip_coin_1_TH",
        alt_dict["amy_flip_coin_1_TH"]
    )
    bob_flip_coin_1_HT = deterministic_action(
        flip_coin_1_HT,
        pre_coin1_H,
        "bob_flip_coin_1_HT",
        alt_dict["bob_flip_coin_1_HT"]
    )
    bob_flip_coin_1_TH = deterministic_action(
        flip_coin_1_TH,
        pre_coin1_T,
        "bob_flip_coin_1_TH",
        alt_dict["bob_flip_coin_1_TH"]
    )
    return {
        "amy_flip_coin_1_HT": amy_flip_coin_1_HT,
        "amy_flip_coin_1_TH": amy_flip_coin_1_TH,
        "bob_flip_coin_1_HT": bob_flip_coin_1_HT,
        "bob_flip_coin_1_TH": bob_flip_coin_1_TH,
    }


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
# New Function 1: compute_alternatives()
# ============================================================

def compute_alternatives(
    action_name: str,
    belief_state: Set[State],
    agent: str,
    model: 'ProbabilisticTwoCoinModel'
) -> Set[str]:
    """
    Compute which action alternatives an agent can distinguish given their belief state.
    
    Key insight: An agent can only distinguish actions if their belief state
    rules out some of the alternative actions' preconditions.
    
    Args:
        action_name: The action that occurred (e.g., "peek_Amy_coin1_H")
        belief_state: Set of states the agent believes are possible
        agent: Agent name ("Amy" or "Bob")
        model: The probabilistic model containing all actions
    
    Returns:
        Set of action names agent cannot distinguish from action_name
    
    Algorithm:
        1. Get static alternatives from action's alt dict
        2. For each alternative action:
           - Check if it can apply to ANY state in agent's belief
           - If yes, include it (agent cannot rule it out)
           - If no, exclude it (agent knows it didn't happen)
        3. Return filtered set
    
    Example:
        action_name = "peek_Amy_coin1_H"
        belief_state = {(1,0), (1,1)}  # Bob knows coin1=H but not coin2
        agent = "Bob"
        
        Static alternatives for Bob: {"peek_Amy_coin1_H", "peek_Amy_coin1_T"}
        
        Check peek_Amy_coin1_H:
          - Precondition: coin1=H
          - Can apply to (1,0)? YES
          - Can apply to (1,1)? YES
          - Include: YES
        
        Check peek_Amy_coin1_T:
          - Precondition: coin1=T
          - Can apply to (1,0)? NO
          - Can apply to (1,1)? NO
          - Include: NO
        
        Return: {"peek_Amy_coin1_H"}
    """
    # Get the action object
    action = model.actions.get(action_name)
    if action is None:
        return {action_name}  # Action not found, return itself
    
    # Get static alternatives for this agent
    if not hasattr(action, 'alt') or agent not in action.alt:
        return {action_name}  # No alternatives defined
    
    static_alternatives = action.alt[agent]
    
    # Filter alternatives based on agent's belief state
    filtered_alternatives = set()
    
    for alt_action_name in static_alternatives:
        # Get the alternative action
        alt_action = model.actions.get(alt_action_name)
        if alt_action is None:
            # Action not found, keep it to be safe
            filtered_alternatives.add(alt_action_name)
            continue
        
        # Check if this alternative action can apply to ANY state in agent's belief
        can_apply = False
        for believed_state in belief_state:
            # Try to apply the action
            successors = alt_action.stochastic_rel(believed_state)
            if successors:  # Non-empty means precondition was satisfied
                can_apply = True
                break
        
        # Include alternative only if it's consistent with agent's beliefs
        if can_apply:
            filtered_alternatives.add(alt_action_name)
    
    # Edge case: if everything was filtered out, return at least the action itself
    if not filtered_alternatives:
        filtered_alternatives.add(action_name)
    
    return filtered_alternatives


# ============================================================
# New Function 2: track_multi_agent_trajectory()  
# ============================================================

def track_multi_agent_trajectory(
    model: 'ProbabilisticTwoCoinModel',
    initial_state: State,
    action_sequence: List[str],
    agents_equiv_classes: Dict[str, List[Set[State]]]
) -> Dict[str, List[tuple]]:
    """
    Track belief state trajectories for multiple agents through an action sequence.
    
    Args:
        model: The probabilistic model with actions
        initial_state: Ground truth initial state (must be a tuple like (1, 0))
        action_sequence: Sequence of action names
        agents_equiv_classes: Dict mapping agent names to their equivalence classes
            Example: {
                "Amy": [{(1,0), (1,1)}, {(0,0), (0,1)}],  # Amy sees coin1
                "Bob": [{(1,0), (0,0)}, {(1,1), (0,1)}]   # Bob sees coin2
            }
    
    Returns:
        Dict mapping agent name to trajectory
    """
    # Type checking
    if not isinstance(initial_state, tuple) or len(initial_state) != 2:
        raise TypeError(f"initial_state must be a 2-tuple (coin1, coin2), got {type(initial_state)}: {initial_state}")
    
    if not all(isinstance(x, int) and x in (0, 1) for x in initial_state):
        raise ValueError(f"initial_state values must be 0 or 1, got {initial_state}")
    
    # Initialize trajectories for each agent
    trajectories = {}
    
    for agent_name, equiv_classes in agents_equiv_classes.items():
        # All agents start believing all states are possible
        initial_belief = set()
        for equiv_class in equiv_classes:
            initial_belief.update(equiv_class)
        
        trajectories[agent_name] = [("initial", initial_belief)]
    
    # Track ground truth state
    current_state = initial_state
    print(f"DEBUG: Starting with current_state = {current_state} (type: {type(current_state)})")
    
    # Process each action
    for action_idx, action_name in enumerate(action_sequence):
        print(f"\nDEBUG: Processing action {action_idx}: {action_name}")
        print(f"DEBUG: current_state before action = {current_state} (type: {type(current_state)})")
        
        action = model.actions.get(action_name)
        if action is None:
            print(f"Warning: Action {action_name} not found in model")
            continue
        
        # Apply action to ground truth (deterministically for this trajectory)
        try:
            successors = model.get_action_successors(current_state, action_name)
        except TypeError as e:
            print(f"ERROR: Failed to get successors for action {action_name} from state {current_state}")
            print(f"  current_state type: {type(current_state)}")
            print(f"  current_state value: {current_state}")
            raise
        
        if not successors:
            print(f"Warning: No successors for action {action_name} from state {current_state}")
            print(f"  This likely means the precondition is not satisfied")
            continue
        
        # Take the most probable successor as ground truth
        next_state = max(successors.items(), key=lambda x: x[1])[0]
        print(f"DEBUG: Ground truth transitions {current_state} -> {next_state}")
        
        # Update each agent's belief
        for agent_name, equiv_classes in agents_equiv_classes.items():
            current_belief = trajectories[agent_name][-1][1]  # Get last belief state
            print(f"DEBUG: Current belief for agent {agent_name}: {current_belief}")
            new_belief = set()
            for alt_action in alt_dict[action_name][agent_name]:
                print(f"DEBUG: Checking alternative action {alt_action}")
                for believed_state in current_belief:
                    # Try to apply the action to this believed state
                    try:
                        state_successors = model.get_action_successors(believed_state, alt_action)
                        
                        if state_successors:
                            # Action can apply - add all possible successors
                            for successor_state in state_successors.keys():
                                new_belief.add(successor_state)
                        # else:
                        #     # Action cannot apply to this state (precondition fails)
                        #     # State remains unchanged
                        #     new_belief.add(believed_state)
                    except Exception as e:
                        print(f"  WARNING: Error applying {alt_action} to {believed_state}: {e}")
                        # On error, assume state unchanged
                        new_belief.add(believed_state)
            print(f"DEBUG: New belief for agent {agent_name}: {new_belief}")
            if not new_belief:
                # If nothing worked, keep current belief
                new_belief = current_belief
            
            print(f"  {agent_name}: {len(current_belief)} states -> {len(new_belief)} states")
            trajectories[agent_name].append((action_name, new_belief))
        current_state = next_state
    
    return trajectories


# ============================================================
# Helper function for pretty printing trajectories
# ============================================================

def print_multi_agent_trajectory(
    trajectories: Dict[str, List[tuple]],
    ground_truth_sequence: List[State] = None
):
    """
    Pretty print multi-agent trajectories.
    
    Args:
        trajectories: Output from track_multi_agent_trajectory()
        ground_truth_sequence: Optional list of actual states at each step
    """
    print("=" * 70)
    print("MULTI-AGENT TRAJECTORY TRACKING")
    print("=" * 70)
    
    if ground_truth_sequence:
        print("\nGround Truth Sequence:")
        for i, state in enumerate(ground_truth_sequence):
            print(f"  Step {i}: {state}")
        print()
    
    for agent_name, trajectory in trajectories.items():
        print(f"\n{agent_name}'s Belief Trajectory:")
        print("-" * 70)
        
        for i, (action, belief_state) in enumerate(trajectory):
            print(f"\nStep {i}: After '{action}'")
            print(f"  Believes: {belief_state}")
            print(f"  # states: {len(belief_state)}")
            
            # Show what agent knows about each coin
            if belief_state:
                coin1_values = {s[0] for s in belief_state}
                coin2_values = {s[1] for s in belief_state}
                
                if len(coin1_values) == 1:
                    val = list(coin1_values)[0]
                    print(f"  Knows: coin1={'H' if val==1 else 'T'}")
                else:
                    print(f"  Uncertain: coin1 could be H or T")
                
                if len(coin2_values) == 1:
                    val = list(coin2_values)[0]
                    print(f"  Knows: coin2={'H' if val==1 else 'T'}")
                else:
                    print(f"  Uncertain: coin2 could be H or T")
    
    print("\n" + "=" * 70)


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
    actions = {}
    actions.update(create_deterministic_announcements())
    actions.update(create_deterministic_peeks())
    actions.update(create_deterministic_flip_coin_1())
    if include_noisy_actions:
        actions.update(create_noisy_peeks(noise_accuracy))
    
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
    model = build_complete_model(include_noisy_actions=False)
    
    # Define equivalence classes
    amy_equiv = [
        {(1,0), (1,1)},  # Amy sees coin1=H
        {(0,0), (0,1)}   # Amy sees coin1=T
    ]
    
    bob_equiv = [
        {(1,0), (0,0)},  # Bob sees coin2=T
        {(1,1), (0,1)}   # Bob sees coin2=H
    ]
    
    agents = {
        "Amy": amy_equiv,
        "Bob": bob_equiv
    }
    
    # Test scenario
    initial = (1, 0)  # coin1=H, coin2=T
    actions = ["peek_Bob_coin1_H", "amy_flip_coin_1_HT"]
    
    # Track trajectories
    trajectories = track_multi_agent_trajectory(model, initial, actions, agents)
    
    # Print results
    print_multi_agent_trajectory(trajectories)
    
    # Test compute_alternatives
    print("\n" + "=" * 70)
    print("TESTING compute_alternatives()")
    print("=" * 70)
    
    # Test 1: Bob knows coin1=H
    bob_belief_knows_H = {(1,0), (1,1)}
    alts = compute_alternatives("peek_Amy_coin1_H", bob_belief_knows_H, "Bob", model)
    print(f"\nTest 1: Bob knows coin1=H")
    print(f"  Bob's belief: {bob_belief_knows_H}")
    print(f"  Static alternatives: {model.actions['peek_Amy_coin1_H'].alt['Bob']}")
    print(f"  Computed alternatives: {alts}")
    print(f"  Expected: Only peek_Amy_coin1_H (since Bob knows coin1 must be H)")
    
    # Verify: Check if peek_Amy_coin1_T can apply
    print(f"\n  Verification:")
    for state in bob_belief_knows_H:
        t_result = model.actions['peek_Amy_coin1_T'].stochastic_rel(state)
        print(f"    peek_Amy_coin1_T on {state}: {t_result} (precondition {'satisfied' if t_result else 'FAILED'})")
    
    # Test 2: Bob doesn't know coin1
    bob_belief_uncertain = {(1,0), (0,0)}
    alts = compute_alternatives("peek_Amy_coin1_H", bob_belief_uncertain, "Bob", model)
    print(f"\nTest 2: Bob uncertain about coin1")
    print(f"  Bob's belief: {bob_belief_uncertain}")
    print(f"  Static alternatives: {model.actions['peek_Amy_coin1_H'].alt['Bob']}")
    print(f"  Computed alternatives: {alts}")
    print(f"  Expected: Both peek_Amy_coin1_H and peek_Amy_coin1_T")
    
    # Verify: Check if both actions can apply
    print(f"\n  Verification:")
    for state in bob_belief_uncertain:
        h_result = model.actions['peek_Amy_coin1_H'].stochastic_rel(state)
        t_result = model.actions['peek_Amy_coin1_T'].stochastic_rel(state)
        print(f"    State {state}:")
        print(f"      peek_Amy_coin1_H: {h_result} ({'YES' if h_result else 'NO'})")
        print(f"      peek_Amy_coin1_T: {t_result} ({'YES' if t_result else 'NO'})")
    
    # Test 3: Bob knows coin1=T
    bob_belief_knows_T = {(0,0), (0,1)}
    alts = compute_alternatives("peek_Amy_coin1_T", bob_belief_knows_T, "Bob", model)
    print(f"\nTest 3: Bob knows coin1=T")
    print(f"  Bob's belief: {bob_belief_knows_T}")
    print(f"  Static alternatives: {model.actions['peek_Amy_coin1_T'].alt['Bob']}")
    print(f"  Computed alternatives: {alts}")
    print(f"  Expected: Only peek_Amy_coin1_T (since Bob knows coin1 must be T)")
