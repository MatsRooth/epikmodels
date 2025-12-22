#!/usr/bin/env python3
"""
Probabilistic Epistemic KAT - Interactive Demo Script

A standalone demo that doesn't require Jupyter.
Run with: python interactive_demo.py

Requirements:
- matplotlib (for visualizations)
- probabilistic_epikat.py in the same directory
"""

import sys
from probabilistic_epikat import *
import matplotlib.pyplot as plt
import numpy as np

# Check dependencies
try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available. Install with: pip install matplotlib")
    print("Visualizations will be skipped.\n")


def print_header(text):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)


def print_subheader(text):
    """Print a formatted subsection header."""
    print("\n" + "-" * 70)
    print(text)
    print("-" * 70)


def demo_1_state_distributions():
    """Demo: State Probability Distributions (Approach 1)"""
    print_header("DEMO 1: STATE PROBABILITY DISTRIBUTIONS")
    
    print("\nCreating different state distributions...")
    
    # Uniform distribution
    uniform_dist = StateDistribution.uniform()
    print_subheader("Uniform Distribution")
    print(uniform_dist)
    print(f"\nP(coin1=H) = {uniform_dist.marginal_coin1()[1]:.3f}")
    print(f"P(coin2=H) = {uniform_dist.marginal_coin2()[1]:.3f}")
    
    # Biased distribution
    biased_dist = StateDistribution({
        (1, 0): 0.7,  # 70% chance coin1=H, coin2=T
        (0, 1): 0.2,  # 20% chance coin1=T, coin2=H
        (1, 1): 0.1   # 10% chance both H
    })
    print_subheader("Biased Distribution (favors coin1=H)")
    print(biased_dist)
    print(f"\nP(coin1=H) = {biased_dist.marginal_coin1()[1]:.3f}")
    print(f"P(coin2=H) = {biased_dist.marginal_coin2()[1]:.3f}")
    
    # Visualize
    if HAS_MATPLOTLIB:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Uniform
        states = [state_to_string(s) for s in uniform_dist.distribution.keys()]
        probs = list(uniform_dist.distribution.values())
        ax1.bar(states, probs, color='steelblue', alpha=0.7)
        ax1.set_xlabel('State')
        ax1.set_ylabel('Probability')
        ax1.set_title('Uniform Distribution')
        ax1.set_ylim([0, 1.0])
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Biased
        states = [state_to_string(s) for s in biased_dist.distribution.keys()]
        probs = list(biased_dist.distribution.values())
        ax2.bar(states, probs, color='coral', alpha=0.7)
        ax2.set_xlabel('State')
        ax2.set_ylabel('Probability')
        ax2.set_title('Biased Distribution')
        ax2.set_ylim([0, 1.0])
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('demo1_state_distributions.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved to: demo1_state_distributions.png")
        plt.close()


def demo_2_stochastic_actions():
    """Demo: Stochastic Actions (Approach 2)"""
    print_header("DEMO 2: STOCHASTIC ACTIONS (NOISY OBSERVATIONS)")
    
    noise_levels = [0.6, 0.8, 0.9, 0.95, 1.0]
    
    print("\nTesting different sensor accuracy levels...")
    results = []
    
    for accuracy in noise_levels:
        actions = create_noisy_peeks(accuracy=accuracy)
        peek_H = actions["peek_Amy_coin1_H_noisy"]
        
        # When coin is actually H, what's probability of observing H?
        actual_state = (1, 0)  # coin1=H
        result_dist = peek_H.stochastic_rel(actual_state)
        p_correct = result_dist.get(actual_state, 0.0)
        results.append((accuracy, p_correct, 1 - p_correct))
        
        print(f"\nAccuracy: {accuracy*100:5.1f}%")
        print(f"  ✓ Correct observation: {p_correct:.3f}")
        print(f"  ✗ Error rate: {(1-p_correct)*100:5.1f}%")
    
    # Visualize
    if HAS_MATPLOTLIB:
        accuracies = [r[0] for r in results]
        corrects = [r[1] for r in results]
        errors = [r[2] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy vs probability
        ax1.plot(accuracies, corrects, 'o-', color='green', 
                label='Correct observation', linewidth=2, markersize=8)
        ax1.plot(accuracies, errors, 'o-', color='red', 
                label='Error', linewidth=2, markersize=8)
        ax1.set_xlabel('Sensor Accuracy')
        ax1.set_ylabel('Probability')
        ax1.set_title('Observation Probability vs Sensor Accuracy')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Error rate
        ax2.plot([a*100 for a in accuracies], [e*100 for e in errors], 
                'o-', color='red', linewidth=2, markersize=8)
        ax2.set_xlabel('Sensor Accuracy (%)')
        ax2.set_ylabel('Error Rate (%)')
        ax2.set_title('Error Rate vs Sensor Accuracy')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('demo2_stochastic_actions.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved to: demo2_stochastic_actions.png")
        plt.close()


def demo_3_epistemic_beliefs():
    """Demo: Epistemic Beliefs (Approach 3)"""
    print_header("DEMO 3: EPISTEMIC BELIEFS (GRADED CONFIDENCE)")
    
    # Create different belief states
    uncertain_belief = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            frozenset([(1, 0), (1, 1)]): 0.5,
            frozenset([(0, 0), (0, 1)]): 0.5,
        }
    )
    
    confident_belief = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            frozenset([(1, 0), (1, 1)]): 0.9,
            frozenset([(0, 0), (0, 1)]): 0.1,
        }
    )
    
    print_subheader("Uncertain Belief (Maximum Entropy)")
    print(uncertain_belief)
    print(f"Entropy: {uncertain_belief.entropy():.3f} bits")
    print(f"Most likely class: {uncertain_belief.most_likely_class()}")
    
    print_subheader("Confident Belief (Low Entropy)")
    print(confident_belief)
    print(f"Entropy: {confident_belief.entropy():.3f} bits")
    print(f"Most likely class: {confident_belief.most_likely_class()}")
    
    # Visualize
    if HAS_MATPLOTLIB:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Uncertain
        classes = [f"Class {i}" for i in range(len(uncertain_belief.belief_distribution))]
        probs = list(uncertain_belief.belief_distribution.values())
        bars1 = ax1.bar(classes, probs, color='purple', alpha=0.7)
        ax1.set_ylabel('Belief Probability')
        ax1.set_title(f'Uncertain Belief\n(Entropy: {uncertain_belief.entropy():.3f} bits)')
        ax1.set_ylim([0, 1.0])
        ax1.grid(axis='y', alpha=0.3)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Confident
        probs = list(confident_belief.belief_distribution.values())
        bars2 = ax2.bar(classes, probs, color='darkviolet', alpha=0.7)
        ax2.set_ylabel('Belief Probability')
        ax2.set_title(f'Confident Belief\n(Entropy: {confident_belief.entropy():.3f} bits)')
        ax2.set_ylim([0, 1.0])
        ax2.grid(axis='y', alpha=0.3)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('demo3_epistemic_beliefs.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved to: demo3_epistemic_beliefs.png")
        plt.close()


def demo_4_trajectory_computation():
    """Demo: Computing Trajectories"""
    print_header("DEMO 4: TRAJECTORY COMPUTATION")
    
    # Build model
    model = build_complete_model(include_noisy_actions=True, noise_accuracy=0.85)
    initial_state = (1, 0)  # coin1=H, coin2=T
    
    # Single observation
    print_subheader("Single Noisy Observation")
    action_sequence = ["peek_Amy_coin1_H_noisy"]
    final_dist = model.compute_trajectory_probability(action_sequence, initial_state)
    print(f"Initial state: {state_to_string(initial_state)}")
    print(f"Action: {action_sequence[0]}")
    print(f"Probability of correct observation: {final_dist.get((1, 0), 0.0):.4f}")
    print(f"Expected: 0.85")
    print(f"Match: {'✓' if abs(final_dist.get((1, 0), 0.0) - 0.85) < 0.01 else '✗'}")
    
    # Multiple observations
    print_subheader("Multiple Observations (Probability Decay)")
    n_observations = 5
    results = []
    
    for n in range(1, n_observations + 1):
        action_sequence = ["peek_Amy_coin1_H_noisy"] * n
        final_dist = model.compute_trajectory_probability(action_sequence, initial_state)
        prob = final_dist.get((1, 0), 0.0)
        expected = 0.85 ** n
        results.append((n, prob, expected))
        print(f"{n} observation(s): P(all correct) = {prob:.6f} (expected: {expected:.6f})")
    
    # Visualize
    if HAS_MATPLOTLIB:
        ns = [r[0] for r in results]
        probs = [r[1] for r in results]
        expected = [r[2] for r in results]
        
        plt.figure(figsize=(10, 5))
        plt.plot(ns, probs, 'o-', label='Computed', markersize=10, linewidth=2)
        plt.plot(ns, expected, 's--', label='Expected (0.85^n)', 
                alpha=0.7, markersize=8, linewidth=2)
        plt.xlabel('Number of Observations')
        plt.ylabel('Probability (all correct)')
        plt.title('Probability Decay with Multiple Noisy Observations')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('demo4_trajectory_computation.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved to: demo4_trajectory_computation.png")
        plt.close()


def demo_5_noise_comparison():
    """Demo: Comparing Different Noise Levels"""
    print_header("DEMO 5: NOISE LEVEL COMPARISON")
    
    noise_levels = [0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    n_obs = 3
    initial_state = (1, 0)
    
    print(f"\nComparing {n_obs} sequential observations at different accuracy levels...")
    results = []
    
    for noise in noise_levels:
        model = build_complete_model(include_noisy_actions=True, noise_accuracy=noise)
        action_seq = ["peek_Amy_coin1_H_noisy"] * n_obs
        final_dist = model.compute_trajectory_probability(action_seq, initial_state)
        prob = final_dist.get((1, 0), 0.0)
        expected = noise ** n_obs
        results.append((noise, prob, expected))
        print(f"Accuracy {noise*100:5.1f}%: P={prob:.4f} (expected: {expected:.4f})")
    
    # Visualize
    if HAS_MATPLOTLIB:
        noises = [r[0] for r in results]
        probs = [r[1] for r in results]
        
        plt.figure(figsize=(10, 5))
        plt.plot(np.array(noises) * 100, probs, 'o-', 
                markersize=10, linewidth=2, color='darkblue')
        plt.xlabel('Sensor Accuracy (%)')
        plt.ylabel(f'P(all {n_obs} observations correct)')
        plt.title(f'Effect of Sensor Accuracy on {n_obs} Sequential Observations')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('demo5_noise_comparison.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved to: demo5_noise_comparison.png")
        plt.close()


def demo_6_natural_language():
    """Demo: Natural Language Interpretation"""
    print_header("DEMO 6: NATURAL LANGUAGE INTERPRETATION")
    
    def confidence_to_language(prob):
        """Convert probability to natural language expression."""
        if prob >= 0.95:
            return "very confident"
        elif prob >= 0.80:
            return "fairly confident"
        elif prob >= 0.60:
            return "somewhat confident"
        elif prob >= 0.40:
            return "uncertain"
        else:
            return "not confident"
    
    print("\nConnecting probabilities to natural language...")
    
    scenarios = [
        (1, 0.95, "perfect sensor"),
        (1, 0.85, "good sensor"),
        (2, 0.85, "two observations with good sensor"),
        (1, 0.70, "noisy sensor"),
        (3, 0.70, "three observations with noisy sensor"),
    ]
    
    for n_obs, accuracy, description in scenarios:
        model = build_complete_model(include_noisy_actions=True, noise_accuracy=accuracy)
        actions = ["peek_Amy_coin1_H_noisy"] * n_obs
        final_dist = model.compute_trajectory_probability(actions, (1, 0))
        prob = final_dist.get((1, 0), 0.0)
        language = confidence_to_language(prob)
        
        print(f"\nScenario: {description}")
        print(f"  Probability: {prob:.4f}")
        print(f'  Natural language: "Amy is {language} the coin is heads"')


def demo_7_hfst_export():
    """Demo: HFST Export and Path Extraction"""
    print_header("DEMO 7: HFST EXPORT AND PATH EXTRACTION")
    
    if not HFST_AVAILABLE:
        print("\n⚠ HFST not available - skipping this demo")
        print("Install HFST to enable weighted FST export and path extraction")
        return
    
    model = build_complete_model(include_noisy_actions=True, noise_accuracy=0.85)
    
    # Export state distribution
    print_subheader("State Distribution FST")
    T_state = export_state_distribution_to_hfst(model.initial_state_dist, add_self_loops=False)
    print(f"States: {T_state.number_of_states()}")
    print(f"Arcs: {T_state.number_of_arcs()}")
    
    paths = extract_paths_safe(T_state, max_cycles=0)
    print(f"\nExtracted {len(paths)} paths:")
    for path, weight in sorted(paths.items(), key=lambda x: x[1])[:5]:
        prob = weight_to_prob(weight)
        print(f"  {path}: P={prob:.4f}")
    
    # Export complete model
    print_subheader("Complete Model FST with Action")
    action_sequence = ["peek_Amy_coin1_H_noisy"]
    T_complete = export_complete_model_to_hfst(
        model,
        action_sequence=action_sequence,
        include_beliefs=False
    )
    print(f"Action sequence: {action_sequence}")
    print(f"States: {T_complete.number_of_states()}")
    print(f"Arcs: {T_complete.number_of_arcs()}")
    
    paths = extract_paths_safe(T_complete, max_cycles=1, max_paths=10)
    if len(paths) > 0:
        print(f"\nExtracted {len(paths)} paths (top 5 by probability):")
        for i, (path, weight) in enumerate(sorted(paths.items(), key=lambda x: x[1])[:5], 1):
            prob = weight_to_prob(weight)
            display_path = path[:60] + "..." if len(path) > 60 else path
            print(f"{i}. P={prob:.4f}: {display_path}")
    else:
        print("\nCould not extract paths (cyclic transducer)")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("PROBABILISTIC EPISTEMIC KAT - INTERACTIVE DEMO".center(70))
    print("=" * 70)
    print(f"\nHFST Available: {HFST_AVAILABLE}")
    print(f"Matplotlib Available: {HAS_MATPLOTLIB}")
    
    if not HAS_MATPLOTLIB:
        print("\nNote: Install matplotlib for visualizations: pip install matplotlib")
    
    input("\nPress Enter to start demos...")
    
    # Run demos
    demos = [
        demo_1_state_distributions,
        demo_2_stochastic_actions,
        demo_3_epistemic_beliefs,
        demo_4_trajectory_computation,
        demo_5_noise_comparison,
        demo_6_natural_language,
        demo_7_hfst_export,
    ]
    
    for i, demo in enumerate(demos, 1):
        demo()
        if i < len(demos):
            input("\nPress Enter for next demo...")
    
    # Summary
    print_header("DEMO COMPLETE")
    print("\nGenerated files:")
    if HAS_MATPLOTLIB:
        files = [
            "demo1_state_distributions.png",
            "demo2_stochastic_actions.png",
            "demo3_epistemic_beliefs.png",
            "demo4_trajectory_computation.png",
            "demo5_noise_comparison.png",
        ]
        for f in files:
            print(f"  ✓ {f}")
    else:
        print("  (No visualizations - matplotlib not available)")
    
    print("\nKey takeaways:")
    print("  • Three probability types model different aspects of uncertainty")
    print("  • State distributions: uncertainty about initial conditions")
    print("  • Stochastic actions: noisy observations and uncertain outcomes")
    print("  • Epistemic beliefs: graded agent confidence")
    print("  • Probability compounds: 0.85 × 0.85 = 0.72 for two observations")
    print("  • HFST export enables efficient FST operations")
    print("  • Natural language connects to probability thresholds")
    
    print("\nNext steps:")
    print("  • Experiment with different noise levels")
    print("  • Try longer action sequences")
    print("  • Modify the code for your own scenarios")
    print("  • Explore HFST composition for complex models")


if __name__ == "__main__":
    main()