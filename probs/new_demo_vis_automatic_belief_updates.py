#!/usr/bin/env python3
"""
Corrected Epistemic FST Demo - Using Augmented State Actions
"""

from probabilistic_epikat import *
from demo_automatic_belief_updates import update_belief_simple
from epistemic_guarded_strings import *
import os

def create_noisy_peek_with_obs_states(
    coin_index: int,
    agent: str = "Amy",
    accuracy: float = 0.9
) -> ProbabilisticAction:
    """
    Create noisy peek with augmented state space.
    
    State space: (coin1, coin2, last_obs) where last_obs in {"H", "T", None}
    """
    name = f"peek_{agent}_coin{coin_index+1}"
    
    def stochastic_rel(s: State) -> Dict[tuple, float]:
        # Extract base state (might already have observation component)
        if len(s) == 2:
            coin1, coin2 = s
        else:
            coin1, coin2 = s[:2]
        
        base_state = (coin1, coin2)
        true_value = base_state[coin_index]
        
        if true_value == 1:  # Coin is H
            # Observe H with accuracy, T with 1-accuracy
            return {
                (*base_state, "H"): accuracy,
                (*base_state, "T"): 1 - accuracy
            }
        else:  # Coin is T
            return {
                (*base_state, "T"): accuracy,
                (*base_state, "H"): 1 - accuracy
            }
    
    return ProbabilisticAction(
        name=name,
        stochastic_rel=stochastic_rel,
        alt={agent: {name}}
    )


def demo():
    """Corrected demo using augmented-state actions."""
    print("="*70)
    print("EPISTEMIC GUARDED STRING FSTs (CORRECTED)")
    print("="*70)
    print("\nUsing augmented state space with observations...")
    
    if not os.path.exists('epistemic_fsts_corrected'):
        os.makedirs('epistemic_fsts_corrected')
    os.chdir('epistemic_fsts_corrected')
    
    # Setup
    initial_dist = StateDistribution({
        (1, 0): 0.25,
        (0, 1): 0.25,
        (1, 1): 0.25,
        (0, 0): 0.25,
    })
    
    amy_equiv_classes = [
        frozenset({(1, 0), (1, 1)}),  # coin1=H
        frozenset({(0, 0), (0, 1)})   # coin1=T
    ]
    
    amy_initial = EpistemicBeliefState(
        agent="Amy",
        belief_distribution={
            amy_equiv_classes[0]: 0.5,
            amy_equiv_classes[1]: 0.5
        }
    )
    
    # Build model with AUGMENTED-STATE peek action
    model = build_complete_model(include_noisy_actions=False)  # Don't use old actions
    
    # Add the new augmented-state peek action
    peek_Amy_coin1 = create_noisy_peek_with_obs_states(
        coin_index=0, 
        agent="Amy", 
        accuracy=0.9
    )
    model.actions["peek_Amy_coin1"] = peek_Amy_coin1
    
    print("\n✓ Created augmented-state peek action")
    print(f"  Action name: peek_Amy_coin1")
    print(f"  Test from (1,0):")
    test_result = peek_Amy_coin1.stochastic_rel((1, 0))
    for state, prob in test_result.items():
        print(f"    -> {state} with P={prob:.2f}")
    
    # Example 1: Initial state only
    print("\n1. Initial state with beliefs")
    
    T1 = export_epistemic_guarded_string(
        initial_dist,
        {"Amy": amy_initial},
        [],  # No actions yet
        model,
        {"Amy": amy_equiv_classes},
        {"Amy": 0.9},
        n_coins=2
    )
    
    print(f"   States: {T1.number_of_states()}, Arcs: {T1.number_of_arcs()}")
    visualize_epistemic_fst(T1, "epistemic_1_initial", show_beliefs=True)
    print("   → Shows initial beliefs on each world branch")
    
    # Example 2: One observation with belief update
    print("\n2. After one peek (automatic belief update!)")
    print("   Using augmented-state action: peek_Amy_coin1") 
    T2 = export_epistemic_guarded_string(
        initial_dist,
        {"Amy": amy_initial},
        ["peek_Amy_coin1"],  # Use the augmented-state action!
        model,
        {"Amy": amy_equiv_classes},
        {"Amy": 0.9},
        n_coins=2
    )
    
    print(f"   States: {T2.number_of_states()}, Arcs: {T2.number_of_arcs()}")
    visualize_epistemic_fst(T2, "epistemic_2_one_peek_CORRECTED", show_beliefs=True)
    print("   → Shows AUTOMATIC belief update after observation")
    print("   → Different paths show different belief updates")
    
    # Example 3: Two observations
    print("\n3. After two peeks (compounding belief updates)")
    
    T3 = export_epistemic_guarded_string(
        initial_dist,
        {"Amy": amy_initial},
        ["peek_Amy_coin1", "peek_Amy_coin1"],  # Two augmented-state actions
        model,
        {"Amy": amy_equiv_classes},
        {"Amy": 0.9},
        n_coins=2
    )
    
    print(f"   States: {T3.number_of_states()}, Arcs: {T3.number_of_arcs()}")
    visualize_epistemic_fst(T3, "epistemic_3_two_peeks_CORRECTED", show_beliefs=True)
    print("   → Shows belief compounding through multiple observations")
    
    os.chdir('..')
    
    print("\n" + "="*70)
    print("CORRECTED EPISTEMIC FST VISUALIZATION COMPLETE")
    print("="*70)
    print("\nCreated epistemic FSTs in epistemic_fsts_corrected/:")
    print("  • epistemic_1_initial.png - Beliefs on initial branches")
    print("  • epistemic_2_one_peek_CORRECTED.png - NOW WITH DIFFERENT BELIEFS!")
    print("  • epistemic_3_two_peeks_CORRECTED.png - Compounding updates")
    print("\nKey difference:")
    print("  ✗ Old: peek_Amy_coin1_H_noisy (preconditioned, no observation state)")
    print("  ✓ New: peek_Amy_coin1 (augmented state, returns (s, obs))")
    print("\nNow you should see different final beliefs (0.90 and 0.10)!")


if __name__ == "__main__":
    demo()