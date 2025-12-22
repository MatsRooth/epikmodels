#!/usr/bin/env python3
"""
Examples demonstrating all three probabilistic approaches in Epistemic KAT.

Example 1: State Probability (Approach 1)
  - Uncertainty about which world we're in
  - Initial distributions, marginalization

Example 2: Stochastic Actions (Approach 2)
  - Noisy observations with 90% accuracy
  - Probabilistic outcomes

Example 3: Epistemic Beliefs (Approach 3)
  - Agent confidence about knowledge
  - Bayesian updates

Example 4: All Three Combined
  - Full probabilistic epistemic dynamics
  - HFST export and visualization
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from probabilistic_epikat import *


def example1_state_probability():
    """
    Example 1: State Probability Distribution (Approach 1)
    
    Scenario: We don't know the initial state of the coins.
    Model this as a probability distribution over states.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: State Probability (Approach 1)")
    print("=" * 70)
    
    # Scenario 1: Uniform uncertainty
    print("\n--- Scenario 1: Complete Uncertainty (Uniform) ---")
    uniform_dist = StateDistribution.uniform()
    print(f"Distribution: {uniform_dist.distribution}")
    print(f"P(coin1=H) = {uniform_dist.marginal_coin1()[1]:.3f}")
    print(f"P(coin2=H) = {uniform_dist.marginal_coin2()[1]:.3f}")
    
    # Scenario 2: Biased toward coin1=H
    print("\n--- Scenario 2: Biased toward coin1=Heads ---")
    biased_dist = StateDistribution({
        (1, 0): 0.35,  # H,T
        (1, 1): 0.35,  # H,H
        (0, 0): 0.15,  # T,T
        (0, 1): 0.15,  # T,H
    })
    print(f"Distribution:")
    for state, prob in sorted(biased_dist.distribution.items()):
        print(f"  {state}: {prob:.3f}")
    print(f"P(coin1=H) = {biased_dist.marginal_coin1()[1]:.3f}")
    print(f"P(coin2=H) = {biased_dist.marginal_coin2()[1]:.3f}")
    
    # Scenario 3: Known coin1, unknown coin2
    print("\n--- Scenario 3: Known coin1=H, Unknown coin2 ---")
    partial_dist = StateDistribution({
        (1, 0): 0.5,  # H,T
        (1, 1): 0.5,  # H,H
    })
    print(f"Distribution:")
    for state, prob in sorted(partial_dist.distribution.items()):
        print(f"  {state}: {prob:.3f}")
    print(f"P(coin1=H) = {partial_dist.marginal_coin1()[1]:.3f}")
    print(f"P(coin2=H) = {partial_dist.marginal_coin2()[1]:.3f}")
    
    if HFST_AVAILABLE:
        print("\n--- HFST Export ---")
        T = export_state_distribution_to_hfst(uniform_dist)
        print(f"Exported to HFST transducer: {T.number_of_states()} states, {T.number_of_arcs()} arcs")


def example2_stochastic_actions():
    """
    Example 2: Stochastic Actions (Approach 2)
    
    Scenario: Amy tries to observe coin1, but her observation is noisy (90% accurate).
    Model the observation as a stochastic action with two possible outcomes.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Stochastic Actions (Approach 2)")
    print("=" * 70)
    
    # Create noisy peek actions
    noisy_actions = create_noisy_peeks(accuracy=0.9)
    
    print("\n--- Scenario 1: Amy peeks coin1 (90% accurate) ---")
    print("True state: coin1=H, coin2=T")
    
    current_state = (1, 0)  # H, T
    action_H = noisy_actions["peek_Amy_coin1_H_noisy"]
    action_T = noisy_actions["peek_Amy_coin1_T_noisy"]
    
    # Get stochastic outcomes
    outcomes_H = action_H.stochastic_rel(current_state)
    outcomes_T = action_T.stochastic_rel(current_state)
    
    print(f"\nObservation outcomes from state {current_state}:")
    print(f"  Sees 'H' (correct): prob = {outcomes_H.get(current_state, 0.0):.3f}")
    print(f"  Sees 'T' (wrong):   prob = {outcomes_T.get(current_state, 0.0):.3f}")
    
    print("\n--- Scenario 2: Different accuracy levels ---")
    for accuracy in [0.7, 0.8, 0.9, 0.95, 1.0]:
        actions = create_noisy_peeks(accuracy=accuracy)
        action = actions["peek_Amy_coin1_H_noisy"]
        prob_correct = action.stochastic_rel((1, 0)).get((1, 0), 0.0)
        print(f"  Accuracy {accuracy:.0%}: P(correct observation) = {prob_correct:.3f}")
    
    # Show all possible observations
    print("\n--- Scenario 3: All states and observations ---")
    action_H = noisy_actions["peek_Amy_coin1_H_noisy"]
    action_T = noisy_actions["peek_Amy_coin1_T_noisy"]
    
    print("\nWhen coin1=H:")
    for state in [(1, 0), (1, 1)]:
        outcomes_H = action_H.stochastic_rel(state)
        outcomes_T = action_T.stochastic_rel(state)
        print(f"  State {state}: P(sees H)={outcomes_H.get(state, 0.0):.2f}, P(sees T)={outcomes_T.get(state, 0.0):.2f}")
    
    print("\nWhen coin1=T:")
    for state in [(0, 0), (0, 1)]:
        outcomes_H = action_H.stochastic_rel(state)
        outcomes_T = action_T.stochastic_rel(state)
        print(f"  State {state}: P(sees H)={outcomes_H.get(state, 0.0):.2f}, P(sees T)={outcomes_T.get(state, 0.0):.2f}")
    
    if HFST_AVAILABLE:
        print("\n--- HFST Export ---")
        T = export_action_to_hfst(noisy_actions["peek_Amy_coin1_H_noisy"])
        print(f"Exported action to HFST: {T.number_of_states()} states, {T.number_of_arcs()} arcs")


def example3_epistemic_beliefs():
    """
    Example 3: Epistemic Belief Distributions (Approach 3)
    
    Scenario: Amy has uncertain beliefs about which equivalence class contains the true state.
    Model this as a probability distribution over equivalence classes.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Epistemic Beliefs (Approach 3)")
    print("=" * 70)
    
    # Amy's equivalence classes: partition by coin1
    print("\n--- Amy's Epistemic Partition ---")
    print("Amy sees coin1 exactly:")
    for equiv_class in Amy_rel.classes:
        print(f"  {sorted(equiv_class)}")
    
    # Scenario 1: Uniform beliefs
    print("\n--- Scenario 1: Uniform Beliefs ---")
    uniform_beliefs = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            frozenset([(1, 0), (1, 1)]): 0.5,  # 50% sure coin1=H
            frozenset([(0, 0), (0, 1)]): 0.5,  # 50% sure coin1=T
        }
    )
    print(f"Amy's beliefs: {uniform_beliefs.belief_distribution}")
    print(f"Entropy: {uniform_beliefs.entropy():.3f} bits")
    most_likely, prob = uniform_beliefs.most_likely_class()
    print(f"Most likely class: {sorted(most_likely)} with probability {prob:.3f}")
    
    # Scenario 2: Confident beliefs
    print("\n--- Scenario 2: Confident Beliefs ---")
    confident_beliefs = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            frozenset([(1, 0), (1, 1)]): 0.9,  # 90% sure coin1=H
            frozenset([(0, 0), (0, 1)]): 0.1,  # 10% sure coin1=T
        }
    )
    print(f"Amy's beliefs: {confident_beliefs.belief_distribution}")
    print(f"Entropy: {confident_beliefs.entropy():.3f} bits")
    most_likely, prob = confident_beliefs.most_likely_class()
    print(f"Most likely class: {sorted(most_likely)} with probability {prob:.3f}")
    
    # Scenario 3: Bayesian update
    print("\n--- Scenario 3: Bayesian Update ---")
    print("Prior: Uniform beliefs")
    print("Observation: Noisy signal suggesting coin1=H (90% reliable)")
    
    # Observation likelihood: if coin1=H, 90% chance of signal
    observation_likelihood = {
        frozenset([(1, 0), (1, 1)]): 0.9,  # P(signal | coin1=H)
        frozenset([(0, 0), (0, 1)]): 0.1,  # P(signal | coin1=T)
    }
    
    posterior = bayesian_update(uniform_beliefs, observation_likelihood)
    print(f"Posterior beliefs: {posterior.belief_distribution}")
    print(f"Entropy reduced from {uniform_beliefs.entropy():.3f} to {posterior.entropy():.3f} bits")
    
    if HFST_AVAILABLE:
        print("\n--- HFST Export ---")
        T = export_epistemic_beliefs_to_hfst(confident_beliefs, Amy_rel)
        print(f"Exported beliefs to HFST: {T.number_of_states()} states, {T.number_of_arcs()} arcs")


def example4_combined():
    """
    Example 4: All Three Approaches Combined
    
    Scenario: Complete probabilistic epistemic dynamics
    - Initial state uncertainty (Approach 1)
    - Noisy action outcomes (Approach 2)  
    - Agent belief updates (Approach 3)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: All Three Approaches Combined")
    print("=" * 70)
    
    # Build complete model
    model = build_complete_model(include_noisy_actions=True, noise_accuracy=0.9)
    
    print("\n--- Initial Setup ---")
    print(f"State distribution: uniform over {len(model.initial_state_dist.distribution)} states")
    print(f"Amy's beliefs: {len(model.agent_beliefs['Amy'].belief_distribution)} equivalence classes")
    print(f"Bob's beliefs: {len(model.agent_beliefs['Bob'].belief_distribution)} equivalence classes")
    
    # Scenario: Amy performs noisy observation
    print("\n--- Scenario: Amy observes coin1 (noisy) ---")
    
    # Start from specific state
    initial_state = (1, 0)  # coin1=H, coin2=T
    print(f"True initial state: {initial_state}")
    
    # Action sequence
    action_sequence = ["peek_Amy_coin1_H_noisy"]
    print(f"Action: {action_sequence[0]}")
    
    # Compute trajectory probabilities
    final_dist = model.compute_trajectory_probability(action_sequence, initial_state)
    
    print(f"\nFinal state distribution:")
    for state, prob in sorted(final_dist.items(), key=lambda x: -x[1]):
        print(f"  {state}: {prob:.3f}")
    
    # Multiple observations
    print("\n--- Multiple Observations ---")
    initial_state = (1, 0)
    action_sequence = ["peek_Amy_coin1_H_noisy", "peek_Amy_coin1_H_noisy"]
    print(f"True initial state: {initial_state}")
    print(f"Actions: {action_sequence}")
    
    final_dist = model.compute_trajectory_probability(action_sequence, initial_state)
    print(f"\nFinal state distribution:")
    for state, prob in sorted(final_dist.items(), key=lambda x: -x[1]):
        print(f"  {state}: {prob:.3f}")
    
    if HFST_AVAILABLE:
        print("\n--- HFST Export: Complete Model ---")
        
        # Export with single action
        T_single = export_complete_model_to_hfst(
            model,
            action_sequence=["peek_Amy_coin1_H_noisy"],
            include_beliefs=False
        )
        print(f"Single action model: {T_single.number_of_states()} states, {T_single.number_of_arcs()} arcs")
        
        # Export with action sequence
        T_sequence = export_complete_model_to_hfst(
            model,
            action_sequence=["peek_Amy_coin1_H_noisy", "peek_Amy_coin1_H_noisy"],
            include_beliefs=False
        )
        print(f"Two-action sequence: {T_sequence.number_of_states()} states, {T_sequence.number_of_arcs()} arcs")
        
        # Export with beliefs
        T_with_beliefs = export_complete_model_to_hfst(
            model,
            action_sequence=["peek_Amy_coin1_H_noisy"],
            include_beliefs=True
        )
        print(f"With epistemic beliefs: {T_with_beliefs.number_of_states()} states, {T_with_beliefs.number_of_arcs()} arcs")
        
        print("\n--- Extract Paths ---")
        print("Note: FSTs may have cycles (self-loops), limiting path extraction")
        print("Use extract_paths_safe() for cyclic transducers:")
        print("  paths = extract_paths_safe(T, max_cycles=1)")
        print("Or use composition for complex sequences")


def example5_comparison():
    """
    Example 5: Comparing Different Probability Types
    
    Show how the three approaches capture different aspects of uncertainty.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Comparing Probability Types")
    print("=" * 70)
    
    print("\n--- Question: What is the probability that coin1 is Heads? ---\n")
    
    # Approach 1: State probability
    print("APPROACH 1 (State Probability):")
    print("  Interpretation: Objective uncertainty about the world")
    state_dist = StateDistribution({
        (1, 0): 0.3,
        (1, 1): 0.2,
        (0, 0): 0.3,
        (0, 1): 0.2,
    })
    p_coin1_h = state_dist.marginal_coin1()[1]
    print(f"  P(coin1=H) = {p_coin1_h:.3f}")
    print("  Meaning: Before we look, 50% chance coin1 is actually Heads")
    
    # Approach 2: Action probability
    print("\nAPPROACH 2 (Action Probability):")
    print("  Interpretation: Uncertainty about observation outcomes")
    noisy_actions = create_noisy_peeks(accuracy=0.9)
    action = noisy_actions["peek_Amy_coin1_H_noisy"]
    p_correct = action.stochastic_rel((1, 0)).get((1, 0), 0.0)
    print(f"  P(observe 'H' | coin1=H) = {p_correct:.3f}")
    print("  Meaning: If coin1 is Heads, 90% chance we see 'H'")
    
    # Approach 3: Epistemic belief
    print("\nAPPROACH 3 (Epistemic Belief):")
    print("  Interpretation: Agent's subjective confidence")
    beliefs = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            frozenset([(1, 0), (1, 1)]): 0.7,
            frozenset([(0, 0), (0, 1)]): 0.3,
        }
    )
    p_belief = beliefs.belief_distribution[frozenset([(1, 0), (1, 1)])]
    print(f"  P(Amy believes coin1=H) = {p_belief:.3f}")
    print("  Meaning: Amy is 70% confident coin1 is Heads")
    
    print("\n--- Key Differences ---")
    print("1. Approach 1: Models objective uncertainty about reality")
    print("2. Approach 2: Models stochastic processes and noisy observations")
    print("3. Approach 3: Models subjective beliefs and confidence levels")
    print("\nAll three can coexist in the same model!")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("PROBABILISTIC EPISTEMIC KAT - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    if not HFST_AVAILABLE:
        print("\nWARNING: HFST not available. Install hfst-python for full functionality.")
        print("Examples will run but HFST export will be skipped.\n")
    
    example1_state_probability()
    example2_stochastic_actions()
    example3_epistemic_beliefs()
    example4_combined()
    example5_comparison()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    
    if HFST_AVAILABLE:
        print("\nNext steps:")
        print("1. Use export functions to generate HFST transducers")
        print("2. Visualize with T.view() or save with write_to_file()")
        print("3. Compose transducers to model complex scenarios")
        print("4. Use extract_paths() to enumerate trajectories")
        print("5. Use n_best() to find most likely paths")


if __name__ == "__main__":
    main()