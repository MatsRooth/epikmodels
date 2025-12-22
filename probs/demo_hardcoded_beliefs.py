#!/usr/bin/env python3
"""
Epistemic Belief Visualization Demo

Demonstrates:
1. Agent's initial belief state
2. Belief update after observation
3. Comparing beliefs of multiple agents
4. Entropy changes through observations

Following Campbell & Rooth's epistemic semantics.
"""

from probabilistic_epikat import *
from demo_guarded_strings import export_guarded_string_automaton
from clean_fst_vis import view_fst_clean
import os


def visualize_belief_state(belief_state: EpistemicBeliefState, filename: str, title: str):
    """
    Visualize an agent's belief state as a simple weighted graph.
    
    Each equivalence class becomes a node, weighted by belief probability.
    """
    if not HFST_AVAILABLE:
        print("HFST not available")
        return
    
    tr = hfst.HfstBasicTransducer()
    
    # Create a node for each equivalence class
    tr.add_state(0)  # Start state
    
    for idx, (equiv_class, prob) in enumerate(belief_state.beliefs.items(), 1):
        tr.add_state(idx)
        
        # Label with representative state from class
        rep_state = sorted(equiv_class)[0]
        class_label = f"class_{state_to_bits(rep_state)}"
        
        # Transition from start to this class with belief weight
        weight = prob_to_weight(prob)
        tr.add_transition(0, idx, class_label, class_label, weight)
        tr.set_final_weight(idx, 0.0)
    
    T = hfst.HfstTransducer(tr)
    view_fst_clean(T, filename, ".")
    print(f"   Entropy: {belief_state.entropy():.3f} bits")
    print(f"   Most likely: {belief_state.most_likely_class()}")


def state_to_bits(state):
    """Convert state to bit string."""
    return ''.join(str(b) for b in state)


def demo_beliefs():
    """Demonstrate epistemic beliefs and updates."""
    print("="*70)
    print("EPISTEMIC BELIEF DEMONSTRATION")
    print("="*70)
    
    if not os.path.exists('belief_demo'):
        os.makedirs('belief_demo')
    os.chdir('belief_demo')
    
    # Setup: Amy can only see coin1, Bob can only see coin2
    amy_equiv_classes = [
        {(1, 0), (1, 1)},  # Amy knows coin1=H but not coin2
        {(0, 0), (0, 1)}   # Amy knows coin1=T but not coin2
    ]
    
    bob_equiv_classes = [
        {(0, 1), (1, 1)},  # Bob knows coin2=H but not coin1
        {(0, 0), (1, 0)}   # Bob knows coin2=T but not coin1
    ]
    
    # Example 1: Uncertain Amy (50-50 about coin1)
    print("\n1. AMY'S INITIAL BELIEF - Uncertain (50-50)")
    print("   Amy doesn't know which coin1 is")
    amy_uncertain = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            frozenset([(1, 0), (1, 1)]): 0.5,
            frozenset([(0, 0), (0, 1)]): 0.5,
        }
    )
    visualize_belief_state(amy_uncertain, "belief_1_amy_uncertain", "Amy Uncertain")
    
    # Example 2: Confident Amy (90-10 about coin1=H)
    print("\n2. AMY'S BELIEF - Confident (90% coin1=H)")
    print("   Amy is fairly confident coin1 is heads")
    amy_confident = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            frozenset([(1, 0), (1, 1)]): 0.9,
            frozenset([(0, 0), (0, 1)]): 0.1,
        }
    )
    visualize_belief_state(amy_confident, "belief_2_amy_confident", "Amy Confident")
    
    # Example 3: Bob's belief (different partition!)
    print("\n3. BOB'S BELIEF - Bob sees different coin")
    print("   Bob knows coin2 but not coin1")
    bob_belief = EpistemicBeliefState(
        agent="Bob",
        belief_distribution={
            frozenset([(0, 1), (1, 1)]): 0.7,
            frozenset([(0, 0), (1, 0)]): 0.3,
        }
    )
    visualize_belief_state(bob_belief, "belief_3_bob_different", "Bob's Belief")
    
    # Example 4: Belief update scenario
    print("\n" + "="*70)
    print("BELIEF UPDATE THROUGH OBSERVATION")
    print("="*70)
    
    # Initial: Amy uncertain
    print("\n4a. BEFORE observation: Amy uncertain")
    initial_belief = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            frozenset([(1, 0), (1, 1)]): 0.5,
            frozenset([(0, 0), (0, 1)]): 0.5,
        }
    )
    visualize_belief_state(initial_belief, "belief_4a_before_obs", "Before Observation")
    
    # After observation with 90% sensor accuracy
    print("\n4b. AFTER observing 'H' (90% accurate sensor)")
    print("   Amy observes H → updates belief about coin1=H")
    
    # Bayesian update:
    # P(coin1=H | obs=H) = P(obs=H | coin1=H) * P(coin1=H) / P(obs=H)
    #                     = 0.9 * 0.5 / (0.9*0.5 + 0.1*0.5)
    #                     = 0.45 / 0.5 = 0.9
    
    updated_belief = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            frozenset([(1, 0), (1, 1)]): 0.9,
            frozenset([(0, 0), (0, 1)]): 0.1,
        }
    )
    visualize_belief_state(updated_belief, "belief_4b_after_obs", "After Observation")
    
    print(f"\n   Entropy decreased: {initial_belief.entropy():.3f} → {updated_belief.entropy():.3f}")
    print(f"   Confidence increased: 50% → 90%")
    
    # Example 5: Multiple observations compound
    print("\n5. AFTER TWO observations of 'H' (90% accurate)")
    print("   Second observation increases confidence further")
    
    # Second Bayesian update:
    # P(coin1=H | two obs=H) ≈ 0.988
    highly_confident = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            frozenset([(1, 0), (1, 1)]): 0.988,
            frozenset([(0, 0), (0, 1)]): 0.012,
        }
    )
    visualize_belief_state(highly_confident, "belief_5_two_obs", "After Two Observations")
    
    print(f"\n   Entropy: {highly_confident.entropy():.3f} (nearly certain!)")
    
    os.chdir('..')
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nCreated belief visualizations in belief_demo/:")
    print("  1. Amy uncertain (50-50) - High entropy")
    print("  2. Amy confident (90-10) - Low entropy")
    print("  3. Bob's belief - Different equivalence classes!")
    print("  4a/b. Belief update - Before/after observation")
    print("  5. Multiple observations - Compounding confidence")
    print("\nKey insights:")
    print("  • Beliefs are distributions over equivalence classes")
    print("  • Different agents have different partitions")
    print("  • Observations update beliefs (Bayesian)")
    print("  • Entropy measures uncertainty")
    print("  • Multiple observations compound confidence")


def demo_belief_action_interaction():
    """
    Demonstrate how beliefs interact with actions in guarded strings.
    
    This shows the full epistemic scenario:
    - Initial world state (objective)
    - Agent's belief about state (subjective)
    - Action taken
    - Resulting world state
    - Updated belief
    """
    print("\n" + "="*70)
    print("BELIEF + ACTION INTERACTION")
    print("="*70)
    
    if not os.path.exists('belief_action_demo'):
        os.makedirs('belief_action_demo')
    os.chdir('belief_action_demo')
    
    # Scenario: Amy is uncertain, then peeks at coin1
    print("\nScenario: Amy uncertain → peeks at coin1 → becomes confident")
    
    # Initial state distribution (objective reality)
    initial_dist = StateDistribution({
        (1, 0): 0.3,  # Reality: probably coin1=H
        (1, 1): 0.3,
        (0, 1): 0.2,
        (0, 0): 0.2,
    })
    
    # Amy's initial belief (subjective - she's uncertain)
    amy_initial = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            frozenset([(1, 0), (1, 1)]): 0.5,
            frozenset([(0, 0), (0, 1)]): 0.5,
        }
    )
    print(f"\n1. INITIAL STATE:")
    print(f"   Objective world: P(coin1=H) = {initial_dist.marginal_probability(0, 1):.1f}")
    print(f"   Amy's belief: P(coin1=H) = 0.5 (uncertain)")
    print(f"   Amy's entropy: {amy_initial.entropy():.3f} bits")
    
    # Action: Amy peeks with 90% accuracy
    model = build_complete_model(include_noisy_actions=True, noise_accuracy=0.9)
    
    # Create guarded string showing the action
    T1 = export_guarded_string_automaton(
        initial_dist,
        ["peek_Amy_coin1_H_noisy"],
        model,
        n_coins=2
    )
    print(f"\n2. ACTION: Amy peeks at coin1 (90% accurate)")
    print(f"   FST States: {T1.number_of_states()}, Arcs: {T1.number_of_arcs()}")
    view_fst_clean(T1, "belief_action_1_peek", ".")
    print(f"   → Graph shows branching: correct obs (90%) vs error (10%)")
    
    # Amy's updated belief after observing "H"
    # If she sees "H", she becomes 90% confident coin1=H
    amy_after_obs = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            frozenset([(1, 0), (1, 1)]): 0.9,
            frozenset([(0, 0), (0, 1)]): 0.1,
        }
    )
    
    print(f"\n3. AFTER OBSERVATION (observed 'H'):")
    print(f"   Amy's updated belief: P(coin1=H) = 0.9")
    print(f"   Amy's entropy: {amy_after_obs.entropy():.3f} bits (reduced!)")
    print(f"   Confidence gain: {amy_initial.entropy() - amy_after_obs.entropy():.3f} bits")
    
    # Visualize the belief update
    visualize_belief_state(amy_initial, "belief_action_2_before", "Amy Before")
    visualize_belief_state(amy_after_obs, "belief_action_3_after", "Amy After")
    
    # Example with announcement (perfect knowledge)
    print("\n4. COMPARISON: Announcement (perfect knowledge)")
    print("   If someone announces 'coin1 is H' instead of Amy peeking...")
    
    # After announcement, Amy knows for certain
    amy_after_announce = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            frozenset([(1, 0), (1, 1)]): 1.0,
            frozenset([(0, 0), (0, 1)]): 0.0,
        }
    )
    
    visualize_belief_state(amy_after_announce, "belief_action_4_announce", "After Announcement")
    print(f"   Amy's belief: P(coin1=H) = 1.0 (certain)")
    print(f"   Amy's entropy: {amy_after_announce.entropy():.3f} bits (zero!)")
    
    os.chdir('..')
    
    print("\n" + "="*70)
    print("BELIEF-ACTION INTERACTION SUMMARY")
    print("="*70)
    print("\nCreated visualizations in belief_action_demo/:")
    print("  • belief_action_1_peek.png - Guarded string with noisy peek")
    print("  • belief_action_2_before.png - Amy's belief before")
    print("  • belief_action_3_after.png - Amy's belief after (90% confident)")
    print("  • belief_action_4_announce.png - After announcement (100% certain)")
    print("\nKey insights:")
    print("  • Actions change both world state AND beliefs")
    print("  • Noisy observations → partial belief update")
    print("  • Perfect knowledge → complete belief update")
    print("  • Entropy quantifies information gained")


def main():
    """Run all epistemic belief demos."""
    demo_beliefs()
    demo_belief_action_interaction()
    
    print("\n" + "="*70)
    print("ALL EPISTEMIC DEMOS COMPLETE!")
    print("="*70)
    print("\nYou now have visualizations of:")
    print("  ✓ Agent belief states (distributions over equivalence classes)")
    print("  ✓ Different agents with different knowledge partitions")
    print("  ✓ Belief updates through observations")
    print("  ✓ Interaction between beliefs and actions")
    print("  ✓ Entropy changes quantifying information gain")
    print("\nThese demonstrate Campbell & Rooth's epistemic semantics")
    print("extended with probabilistic beliefs (Approach 3)!")


if __name__ == "__main__":
    main()