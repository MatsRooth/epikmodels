#!/usr/bin/env python3
"""
Automatic Bayesian Belief Updates

This module implements automatic belief updates for agents based on:
1. Prior beliefs
2. Observations (actions taken)
3. Sensor accuracy
4. Equivalence classes (what agent can distinguish)

Key functions:
- compute_observation_likelihood() - P(observation | true state)
- update_belief_from_action() - Automatic Bayesian update
- simulate_epistemic_trajectory() - Track beliefs through action sequence
"""

from probabilistic_epikat import *
from typing import Dict, FrozenSet, Tuple, List


def compute_observation_likelihood(
    action: ProbabilisticAction,
    equiv_classes: List[FrozenSet[State]],
    observation_outcome: str
) -> Dict[FrozenSet[State], float]:
    """
    Compute P(observation | equivalence class).
    
    For each equivalence class, compute the probability of seeing
    the observed outcome if the true state is in that class.
    
    Args:
        action: The observation action (e.g., noisy peek)
        equiv_classes: Agent's equivalence classes
        observation_outcome: The observed outcome (action name)
    
    Returns:
        Dictionary mapping each equivalence class to likelihood
    """
    likelihoods = {}
    
    for equiv_class in equiv_classes:
        # For this equivalence class, compute average likelihood
        # across all possible true states in the class
        total_likelihood = 0.0
        
        for true_state in equiv_class:
            # Get the distribution of outcomes from this state
            outcome_dist = action.stochastic_rel(true_state)
            
            # The likelihood is the probability of the observed outcome
            # We need to check if this action would produce the observation
            # This is simplified - assumes observation_outcome in action results
            likelihood = 0.0
            
            for result_state, prob in outcome_dist.items():
                # Check if this result corresponds to observation
                # For peek actions, we observe correctly if state unchanged
                if result_state == true_state:
                    # Correct observation
                    if "H" in observation_outcome and true_state[0] == 1:
                        likelihood += prob
                    elif "T" in observation_outcome and true_state[0] == 0:
                        likelihood += prob
                else:
                    # Error observation
                    if "H" in observation_outcome and true_state[0] == 0:
                        likelihood += prob
                    elif "T" in observation_outcome and true_state[0] == 1:
                        likelihood += prob
            
            total_likelihood += likelihood
        
        # Average over states in equivalence class (uniform prior within class)
        likelihoods[equiv_class] = total_likelihood / len(equiv_class)
    
    return likelihoods


def update_belief_from_action(
    prior_belief: EpistemicBeliefState,
    action: ProbabilisticAction,
    observation_outcome: str,
    equiv_classes: List[FrozenSet[State]]
) -> EpistemicBeliefState:
    """
    Automatically update agent's belief based on observation.
    
    Performs Bayesian update:
    P(class | obs) = P(obs | class) * P(class) / P(obs)
    
    Args:
        prior_belief: Agent's prior belief
        action: The observation action taken
        observation_outcome: What was observed
        equiv_classes: Agent's equivalence classes
    
    Returns:
        Updated belief state
    """
    # Compute likelihoods
    likelihoods = compute_observation_likelihood(
        action, equiv_classes, observation_outcome
    )
    
    # Bayesian update
    posterior = {}
    normalizer = 0.0
    
    for equiv_class in equiv_classes:
        prior_prob = prior_belief.belief_distribution.get(equiv_class, 0.0)
        likelihood = likelihoods.get(equiv_class, 0.0)
        
        posterior[equiv_class] = likelihood * prior_prob
        normalizer += posterior[equiv_class]
    
    # Normalize
    if normalizer > 0:
        for equiv_class in posterior:
            posterior[equiv_class] /= normalizer
    
    return EpistemicBeliefState(
        agent=prior_belief.agent,
        belief_distribution=posterior
    )


def update_belief_simple(
    prior_belief: EpistemicBeliefState,
    equiv_classes: List[FrozenSet[State]],
    observed_class_idx: int,
    sensor_accuracy: float
) -> EpistemicBeliefState:
    """
    Simplified belief update for single observation.
    
    Assumes:
    - Agent observes which equivalence class they're in
    - Sensor has accuracy rate (correct with probability sensor_accuracy)
    
    Args:
        prior_belief: Prior belief distribution
        equiv_classes: List of equivalence classes
        observed_class_idx: Index of observed class
        sensor_accuracy: Probability of correct observation
    
    Returns:
        Updated belief state
    """
    n_classes = len(equiv_classes)
    error_prob = (1 - sensor_accuracy) / (n_classes - 1) if n_classes > 1 else 0
    
    posterior = {}
    normalizer = 0.0
    
    for idx, equiv_class in enumerate(equiv_classes):
        prior_prob = prior_belief.belief_distribution.get(equiv_class, 0.0)
        
        if idx == observed_class_idx:
            # Observed this class - use sensor accuracy
            likelihood = sensor_accuracy
        else:
            # Didn't observe this class - sensor error
            likelihood = error_prob
        
        posterior[equiv_class] = likelihood * prior_prob
        normalizer += posterior[equiv_class]
    
    # Normalize
    if normalizer > 0:
        for equiv_class in posterior:
            posterior[equiv_class] /= normalizer
    
    return EpistemicBeliefState(
        agent=prior_belief.agent,
        belief_distribution=posterior
    )


def simulate_epistemic_trajectory(
    agent: str,
    equiv_classes: List[FrozenSet[State]],
    initial_belief: EpistemicBeliefState,
    observations: List[Tuple[int, float]]
) -> List[Tuple[str, EpistemicBeliefState]]:
    """
    Simulate trajectory of belief updates through multiple observations.
    
    Args:
        agent: Agent name
        equiv_classes: Agent's equivalence classes
        initial_belief: Starting belief
        observations: List of (observed_class_idx, sensor_accuracy) tuples
    
    Returns:
        List of (description, belief_state) tuples showing evolution
    """
    trajectory = [("Initial", initial_belief)]
    current_belief = initial_belief
    
    for obs_idx, (observed_class, accuracy) in enumerate(observations, 1):
        # Update belief
        updated_belief = update_belief_simple(
            current_belief,
            equiv_classes,
            observed_class,
            accuracy
        )
        
        trajectory.append((f"After observation {obs_idx}", updated_belief))
        current_belief = updated_belief
    
    return trajectory


def demo_automatic_updates():
    """Demonstrate automatic belief updates."""
    print("="*70)
    print("AUTOMATIC BAYESIAN BELIEF UPDATES")
    print("="*70)
    
    # Setup Amy's equivalence classes
    amy_equiv_classes = [
        frozenset({(1, 0), (1, 1)}),  # coin1=H
        frozenset({(0, 0), (0, 1)})   # coin1=T
    ]
    
    # Amy starts uncertain
    amy_initial = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            amy_equiv_classes[0]: 0.5,
            amy_equiv_classes[1]: 0.5
        }
    )
    
    print("\n1. INITIAL BELIEF")
    print(f"   Amy's belief: P(coin1=H) = 0.5, P(coin1=T) = 0.5")
    print(f"   Entropy: {amy_initial.entropy():.3f} bits (maximum uncertainty)")
    
    # Single observation
    print("\n2. AFTER ONE OBSERVATION (sees 'H', 90% accurate)")
    amy_after_1 = update_belief_simple(
        amy_initial,
        list(amy_equiv_classes),
        observed_class_idx=0,  # Observed coin1=H class
        sensor_accuracy=0.9
    )
    
    prob_H_1 = amy_after_1.belief_distribution[amy_equiv_classes[0]]
    print(f"   Automatic Bayesian update:")
    print(f"   P(coin1=H | obs='H') = {prob_H_1:.3f}")
    print(f"   Entropy: {amy_after_1.entropy():.3f} bits")
    print(f"   Information gained: {amy_initial.entropy() - amy_after_1.entropy():.3f} bits")
    
    # Second observation
    print("\n3. AFTER TWO OBSERVATIONS (both see 'H', 90% accurate)")
    amy_after_2 = update_belief_simple(
        amy_after_1,
        list(amy_equiv_classes),
        observed_class_idx=0,
        sensor_accuracy=0.9
    )
    
    prob_H_2 = amy_after_2.belief_distribution[amy_equiv_classes[0]]
    print(f"   Automatic Bayesian update:")
    print(f"   P(coin1=H | two obs='H') = {prob_H_2:.3f}")
    print(f"   Entropy: {amy_after_2.entropy():.3f} bits")
    print(f"   Total information gained: {amy_initial.entropy() - amy_after_2.entropy():.3f} bits")
    
    # Trajectory simulation
    print("\n4. SIMULATING FULL TRAJECTORY (5 observations)")
    observations = [
        (0, 0.9),  # See H, 90% accurate
        (0, 0.9),  # See H, 90% accurate
        (0, 0.9),  # See H, 90% accurate
        (0, 0.9),  # See H, 90% accurate
        (0, 0.9),  # See H, 90% accurate
    ]
    
    trajectory = simulate_epistemic_trajectory(
        "Amy",
        list(amy_equiv_classes),
        amy_initial,
        observations
    )
    
    print(f"\n   Belief evolution:")
    for desc, belief in trajectory:
        prob_H = belief.belief_distribution[amy_equiv_classes[0]]
        entropy = belief.entropy()
        print(f"   {desc:25s}: P(H)={prob_H:.4f}, Entropy={entropy:.4f}")
    
    # Different sensor accuracies
    print("\n5. EFFECT OF SENSOR ACCURACY")
    accuracies = [0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    
    print(f"\n   After ONE observation:")
    for acc in accuracies:
        updated = update_belief_simple(amy_initial, list(amy_equiv_classes), 0, acc)
        prob_H = updated.belief_distribution[amy_equiv_classes[0]]
        print(f"   Accuracy {acc*100:4.0f}%: P(H|obs) = {prob_H:.4f}")
    
    # Contradictory evidence
    print("\n6. CONTRADICTORY EVIDENCE")
    print(f"   Amy sees 'H' twice, then sees 'T' once (all 90% accurate)")
    
    belief = amy_initial
    belief = update_belief_simple(belief, list(amy_equiv_classes), 0, 0.9)  # See H
    prob_after_H = belief.belief_distribution[amy_equiv_classes[0]]
    print(f"   After 1st H: P(coin1=H) = {prob_after_H:.4f}")
    
    belief = update_belief_simple(belief, list(amy_equiv_classes), 0, 0.9)  # See H
    prob_after_HH = belief.belief_distribution[amy_equiv_classes[0]]
    print(f"   After 2nd H: P(coin1=H) = {prob_after_HH:.4f}")
    
    belief = update_belief_simple(belief, list(amy_equiv_classes), 1, 0.9)  # See T!
    prob_after_HHT = belief.belief_distribution[amy_equiv_classes[0]]
    print(f"   After seeing T: P(coin1=H) = {prob_after_HHT:.4f}")
    print(f"   → Belief reverted toward uncertainty due to contradictory evidence")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nAutomatic Bayesian updates:")
    print("  ✓ No manual probability calculations needed")
    print("  ✓ Handles multiple observations sequentially")
    print("  ✓ Accounts for sensor accuracy automatically")
    print("  ✓ Tracks entropy (information gain)")
    print("  ✓ Handles contradictory evidence correctly")
    print("\nUsage:")
    print("  updated = update_belief_simple(prior, classes, observed_idx, accuracy)")


if __name__ == "__main__":
    demo_automatic_updates()