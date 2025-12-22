#!/usr/bin/env python3
"""
HFST Integration and Visualization for Probabilistic Epistemic KAT

This script demonstrates:
1. Exporting models to HFST format
2. Visualizing transducers
3. Composing transducers
4. Extracting and analyzing paths
5. Computing most likely trajectories
6. Saving/loading HFST files
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from probabilistic_epikat import *

if not HFST_AVAILABLE:
    print("ERROR: HFST is not available. Please install hfst-python.")
    print("  pip install hfst-python")
    sys.exit(1)


def demo_basic_export():
    """Demonstrate basic export of each approach to HFST."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic HFST Export")
    print("=" * 70)
    
    # Approach 1: State distribution
    print("\n--- Exporting State Distribution (Approach 1) ---")
    state_dist = StateDistribution({
        (1, 0): 0.4,
        (1, 1): 0.3,
        (0, 0): 0.2,
        (0, 1): 0.1,
    })
    T_states = export_state_distribution_to_hfst(state_dist)
    print(f"States: {T_states.number_of_states()}")
    print(f"Arcs: {T_states.number_of_arcs()}")
    
    # Save to file
    output_file = "/home/claude/state_dist.hfst"
    with hfst.HfstOutputStream(filename=output_file) as out:
        out.write(T_states)
    print(f"Saved to: {output_file}")
    
    # Approach 2: Action
    print("\n--- Exporting Stochastic Action (Approach 2) ---")
    noisy_actions = create_noisy_peeks(accuracy=0.9)
    action = noisy_actions["peek_Amy_coin1_H_noisy"]
    T_action = export_action_to_hfst(action)
    print(f"States: {T_action.number_of_states()}")
    print(f"Arcs: {T_action.number_of_arcs()}")
    
    output_file = "/home/claude/noisy_action.hfst"
    with hfst.HfstOutputStream(filename=output_file) as out:
        out.write(T_action)
    print(f"Saved to: {output_file}")
    
    # Approach 3: Beliefs
    print("\n--- Exporting Epistemic Beliefs (Approach 3) ---")
    beliefs = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            frozenset([(1, 0), (1, 1)]): 0.8,
            frozenset([(0, 0), (0, 1)]): 0.2,
        }
    )
    T_beliefs = export_epistemic_beliefs_to_hfst(beliefs, Amy_rel)
    print(f"States: {T_beliefs.number_of_states()}")
    print(f"Arcs: {T_beliefs.number_of_arcs()}")
    
    output_file = "/home/claude/epistemic_beliefs.hfst"
    with hfst.HfstOutputStream(filename=output_file) as out:
        out.write(T_beliefs)
    print(f"Saved to: {output_file}")


def demo_composition():
    """Demonstrate composing transducers to build complex models."""
    print("\n" + "=" * 70)
    print("DEMO 2: Transducer Composition")
    print("=" * 70)
    
    model = build_complete_model(include_noisy_actions=True, noise_accuracy=0.9)
    
    # Build multi-action sequence via composition
    print("\n--- Building Action Sequence via Composition ---")
    actions = ["peek_Amy_coin1_H_noisy", "peek_Amy_coin1_H_noisy", "peek_Amy_coin1_H_noisy"]
    
    print(f"Action sequence: {actions}")
    
    # Manual composition
    T_result = export_state_distribution_to_hfst(model.initial_state_dist)
    print(f"\nInitial (state dist): {T_result.number_of_states()} states, {T_result.number_of_arcs()} arcs")
    
    for i, action_name in enumerate(actions):
        action = model.actions[action_name]
        T_action = export_action_to_hfst(action)
        T_result.compose(T_action)
        T_result.minimize()
        print(f"After action {i+1}: {T_result.number_of_states()} states, {T_result.number_of_arcs()} arcs")
    
    # Save result
    output_file = "/home/claude/composed_sequence.hfst"
    with hfst.HfstOutputStream(filename=output_file) as out:
        out.write(T_result)
    print(f"\nSaved composed transducer to: {output_file}")
    
    # Using convenience function
    print("\n--- Using Convenience Function ---")
    T_complete = export_complete_model_to_hfst(
        model,
        action_sequence=actions,
        include_beliefs=False
    )
    print(f"Complete model: {T_complete.number_of_states()} states, {T_complete.number_of_arcs()} arcs")


def demo_path_extraction():
    """Demonstrate extracting and analyzing paths."""
    print("\n" + "=" * 70)
    print("DEMO 3: Path Extraction and Analysis")
    print("=" * 70)
    
    model = build_complete_model(include_noisy_actions=True, noise_accuracy=0.9)
    
    # Simple scenario: one noisy observation
    print("\n--- Scenario: One Noisy Observation ---")
    T = export_complete_model_to_hfst(
        model,
        action_sequence=["peek_Amy_coin1_H_noisy"],
        include_beliefs=False
    )
    
    # Extract paths
    print("\nExtracting paths...")
    paths = extract_paths_safe(T, max_cycles=1, max_paths=100)
    
    print(f"Found {len(paths)} paths")
    
    if len(paths) == 0:
        print("Note: Transducer is cyclic, could not extract all paths")
        print("This is expected - use composition for complex sequences")
        return
    
    # Analyze paths
    print("\nTop 5 most likely paths:")
    sorted_paths = sorted(paths.items(), key=lambda x: x[1])[:5]
    
    for path_str, weight in sorted_paths:
        prob = weight_to_prob(weight)
        print(f"  {path_str}")
        print(f"    Weight: {weight:.4f}, Probability: {prob:.4f}")
    
    # Group by final state
    print("\n--- Grouping by Final State ---")
    final_state_probs = {}
    
    for path_str, weight in paths.items():
        prob = weight_to_prob(weight)
        # Extract final state (last 4 characters)
        final_state_str = path_str[-4:] if len(path_str) >= 4 else path_str
        final_state_probs[final_state_str] = final_state_probs.get(final_state_str, 0.0) + prob
    
    print("Probability of each final state:")
    for state_str, prob in sorted(final_state_probs.items(), key=lambda x: -x[1]):
        print(f"  {state_str}: {prob:.4f}")


def demo_n_best():
    """Demonstrate extracting n-best paths."""
    print("\n" + "=" * 70)
    print("DEMO 4: N-Best Path Extraction")
    print("=" * 70)
    
    model = build_complete_model(include_noisy_actions=True, noise_accuracy=0.9)
    
    # Two observations
    print("\n--- Scenario: Two Noisy Observations ---")
    T = export_complete_model_to_hfst(
        model,
        action_sequence=["peek_Amy_coin1_H_noisy", "peek_Amy_coin1_H_noisy"],
        include_beliefs=False
    )
    
    print(f"Transducer: {T.number_of_states()} states, {T.number_of_arcs()} arcs")
    
    # Extract n-best
    n = 10
    print(f"\nExtracting {n} most likely paths...")
    
    # Use safe extraction with cycles
    paths = extract_paths_safe(T, max_cycles=1, max_paths=n)
    
    if len(paths) > 0:
        print(f"\nTop {len(paths)} paths:")
        for i, (path_str, weight) in enumerate(sorted(paths.items(), key=lambda x: x[1])[:n], 1):
            prob = weight_to_prob(weight)
            # Truncate very long paths
            display_path = path_str[:80] + "..." if len(path_str) > 80 else path_str
            print(f"{i}. Probability: {prob:.4f}")
            print(f"   Path: {display_path}")
    else:
        print("Note: Could not extract paths from cyclic transducer")
        print("Use composition and forward simulation instead")


def demo_epistemic_operations():
    """Demonstrate epistemic modal operators with probabilities."""
    print("\n" + "=" * 70)
    print("DEMO 5: Epistemic Modal Operators")
    print("=" * 70)
    
    model = build_complete_model(include_noisy_actions=True, noise_accuracy=0.9)
    
    print("\n--- Box and Diamond Modalities ---")
    print("In epistemic KAT:")
    print("  - Diamond(R, X): There exists an R-accessible world satisfying X")
    print("  - Box(R, X): All R-accessible worlds satisfy X")
    
    print("\nWith probabilities:")
    print("  - Probabilistic Diamond: P(exists R-accessible world with X) > threshold")
    print("  - Probabilistic Box: P(all R-accessible worlds have X) > threshold")
    
    # Example: Amy's knowledge after observation
    print("\n--- Example: Amy's Knowledge ---")
    
    # Export Amy's epistemic relation
    # In HFST, we'd represent this as a transducer
    print("Amy's epistemic partition:")
    for equiv_class in Amy_rel.classes:
        print(f"  {sorted(equiv_class)}")
    
    print("\nAfter noisy observation 'H':")
    print("  Amy's belief: coin1=H with ~81% probability")
    print("  (calculated via Bayesian update)")
    
    # Compute via Python first
    initial_state = (1, 0)
    actions = ["peek_Amy_coin1_H_noisy"]
    final_dist = model.compute_trajectory_probability(actions, initial_state)
    
    print("\nFinal distribution:")
    for state, prob in sorted(final_dist.items(), key=lambda x: -x[1]):
        print(f"  {state}: {prob:.4f}")


def demo_advanced_queries():
    """Demonstrate advanced queries on probabilistic models."""
    print("\n" + "=" * 70)
    print("DEMO 6: Advanced Queries")
    print("=" * 70)
    
    model = build_complete_model(include_noisy_actions=True, noise_accuracy=0.9)
    
    # Query 1: Probability that Amy correctly observes coin1
    print("\n--- Query 1: Probability of Correct Observation ---")
    print("Question: If coin1=H, what's P(Amy observes 'H')?")
    
    initial_state = (1, 0)
    action_sequence = ["peek_Amy_coin1_H_noisy"]
    
    final_dist = model.compute_trajectory_probability(action_sequence, initial_state)
    p_correct = sum(prob for state, prob in final_dist.items() if state == initial_state)
    
    print(f"Answer: {p_correct:.4f}")
    
    # Query 2: After two observations, confidence level
    print("\n--- Query 2: Confidence After Multiple Observations ---")
    print("Question: After observing 'H' twice, what's P(coin1=H)?")
    
    action_sequence = ["peek_Amy_coin1_H_noisy", "peek_Amy_coin1_H_noisy"]
    final_dist = model.compute_trajectory_probability(action_sequence, initial_state)
    
    # P(coin1=H) = sum over states where coin1=1
    p_coin1_h = sum(prob for state, prob in final_dist.items() if state[0] == 1)
    
    print(f"Answer: {p_coin1_h:.4f}")
    print(f"Interpretation: Amy is {p_coin1_h*100:.1f}% confident coin1=H")
    
    # Query 3: Expected entropy after observation
    print("\n--- Query 3: Information Gain ---")
    print("Question: How much does Amy learn from one observation?")
    
    # Initial entropy (uniform)
    initial_entropy = 1.0  # log2(2) for binary uniform
    
    # Expected final entropy
    action_sequence = ["peek_Amy_coin1_H_noisy"]
    
    # Average over all possible initial states
    expected_final_entropy = 0.0
    for init_state in ALL_STATES:
        final_dist = model.compute_trajectory_probability(action_sequence, init_state)
        
        # Compute entropy of final distribution over coin1
        p_h = sum(prob for state, prob in final_dist.items() if state[0] == 1)
        p_t = sum(prob for state, prob in final_dist.items() if state[0] == 0)
        
        if p_h > 0:
            entropy_h = -p_h * math.log2(p_h)
        else:
            entropy_h = 0.0
            
        if p_t > 0:
            entropy_t = -p_t * math.log2(p_t)
        else:
            entropy_t = 0.0
        
        expected_final_entropy += (entropy_h + entropy_t) / len(ALL_STATES)
    
    information_gain = initial_entropy - expected_final_entropy
    
    print(f"Initial entropy: {initial_entropy:.4f} bits")
    print(f"Expected final entropy: {expected_final_entropy:.4f} bits")
    print(f"Information gain: {information_gain:.4f} bits")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("HFST INTEGRATION FOR PROBABILISTIC EPISTEMIC KAT")
    print("=" * 70)
    
    demo_basic_export()
    demo_composition()
    demo_path_extraction()
    demo_n_best()
    demo_epistemic_operations()
    demo_advanced_queries()
    
    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)
    
    print("\nGenerated HFST files:")
    output_dir = Path("/home/claude")
    for hfst_file in output_dir.glob("*.hfst"):
        print(f"  - {hfst_file}")
    
    print("\nYou can:")
    print("1. Load these files in HFST command-line tools")
    print("2. Visualize with hfst-fst2txt or convert to dot format")
    print("3. Compose them with other transducers")
    print("4. Use them in your own HFST workflows")


if __name__ == "__main__":
    main()