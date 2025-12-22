#!/usr/bin/env python3
"""
Guarded String FST Export - Following Campbell & Rooth (2021)

This implements FST export that creates LINEAR automata representing
guarded strings, not one-node-per-world-state.

Key idea from Campbell & Rooth:
- A world is a SEQUENCE: initial_state event0 state1 event1 state2 ...
- The FST reads/writes this sequence
- States in FST = positions in the sequence
- Transitions = reading/writing symbols in sequence

Example path in FST:
  q0 --[1]--> q1 --[0]--> q2 --[announceH1]--> q3 --[1]--> q4 --[0]--> q5 (final)
  
This represents the guarded string: "10 announceH1 10"
"""

import sys
try:
    import hfst_dev as hfst
    HFST_AVAILABLE = True
except ImportError:
    try:
        import hfst
        HFST_AVAILABLE = True
    except ImportError:
        HFST_AVAILABLE = False

from probabilistic_epikat import *
import os


def export_guarded_string_automaton(
    initial_dist: StateDistribution,
    action_sequence: List[str],
    model: 'ProbabilisticTwoCoinModel',
    n_coins: int = 2
) -> 'hfst.HfstTransducer':
    """
    Export as guarded string automaton following Campbell & Rooth.
    
    Structure:
    - Each world is a path: s0 a0 s1 a1 s2 ...
    - States encode bit sequences
    - Actions label transitions
    - Probabilities weight paths
    
    Args:
        initial_dist: Initial state distribution
        action_sequence: List of action names
        model: The probabilistic model
        n_coins: Number of coins (determines state width)
    
    Returns:
        HFST transducer where paths = guarded strings
    """
    if not HFST_AVAILABLE:
        raise ImportError("HFST not available")
    
    tr = hfst.HfstBasicTransducer()
    
    # Initial state
    current_node = 0
    tr.add_state(current_node)
    
    # Branch for each initial world state
    # Each branch will be a separate path through the automaton
    node_counter = 1
    
    # Group by initial state
    branches = {}  # world_state -> [(path_history, current_node, cumulative_prob)]
    
    for world_state, init_prob in initial_dist.distribution.items():
        # Create path for this world
        # First, output the initial state bits
        prev_node = 0
        
        # Output bits of initial state
        bits = state_to_bits(world_state)
        for bit_idx, bit in enumerate(bits):
            next_node = node_counter
            node_counter += 1
            tr.add_state(next_node)
            
            # Weight only on last bit
            weight = prob_to_weight(init_prob) if bit_idx == len(bits) - 1 else 0.0
            
            tr.add_transition(prev_node, next_node, bit, bit, weight)
            prev_node = next_node
        
        # Track this branch
        branches[world_state] = [([], prev_node, init_prob)]
    
    # Now apply actions sequentially
    for action_name in action_sequence:
        action = model.actions[action_name]
        new_branches = {}
        
        for world_state, paths in branches.items():
            for path_history, node, cumulative_prob in paths:
                # Apply action to this world state
                if action.stochastic_rel(world_state) is not None:
                    successors = action.stochastic_rel(world_state)
                    
                    for next_state, action_prob in successors.items():
                        # Add action transition
                        action_node = node_counter
                        node_counter += 1
                        tr.add_state(action_node)
                        
                        action_weight = prob_to_weight(action_prob)
                        tr.add_transition(node, action_node, 
                                        action_name, action_name, action_weight)
                        
                        # Add next state bits
                        prev_node = action_node
                        bits = state_to_bits(next_state)
                        for bit in bits:
                            next_node = node_counter
                            node_counter += 1
                            tr.add_state(next_node)
                            tr.add_transition(prev_node, next_node, bit, bit, 0.0)
                            prev_node = next_node
                        
                        # Track new branch
                        new_path_history = path_history + [(action_name, next_state)]
                        new_cumulative = cumulative_prob * action_prob
                        
                        if next_state not in new_branches:
                            new_branches[next_state] = []
                        new_branches[next_state].append(
                            (new_path_history, prev_node, new_cumulative)
                        )
        
        branches = new_branches
    
    # Mark final states
    for world_state, paths in branches.items():
        for path_history, node, cumulative_prob in paths:
            tr.set_final_weight(node, 0.0)
    
    return hfst.HfstTransducer(tr)


def state_to_bits(state):
    """Convert state tuple to bit string."""
    return ''.join(str(b) for b in state)


def demo_guarded_strings():
    """Demonstrate guarded string export."""
    print("="*70)
    print("GUARDED STRING AUTOMATON DEMO")
    print("="*70)
    print("\nFollowing Campbell & Rooth (2021): Worlds as sequential paths")
    
    from clean_fst_vis import view_fst_clean
    
    if not os.path.exists('guarded_strings'):
        os.makedirs('guarded_strings')
    os.chdir('guarded_strings')
    
    # Example 1: Simple initial distribution
    print("\n1. Initial state distribution only")
    dist = StateDistribution({
        (1, 0): 0.4,
        (0, 1): 0.1,
        (1, 1): 0.2,
        (0, 0): 0.3,

    })
    model = build_complete_model()
    
    T1 = export_guarded_string_automaton(dist, [], model, n_coins=2)
    print(f"   States: {T1.number_of_states()}, Arcs: {T1.number_of_arcs()}")
    view_fst_clean(T1, "gs_1_initial_only", ".")
    
    # Example 2: With one deterministic action
    print("\n2. Initial distribution + deterministic announce")
    det_announce = deterministic_action(
        lambda s: (s[0], s[1]),
        lambda s: s[0] == 1,
        "announceH1"
    )
    model.actions["announceH1"] = det_announce
    
    T2 = export_guarded_string_automaton(dist, ["announceH1"], model, n_coins=2)
    print(f"   States: {T2.number_of_states()}, Arcs: {T2.number_of_arcs()}")
    view_fst_clean(T2, "gs_2_with_announce", ".")
    
    # Example 3: With noisy action
    print("\n3. Initial distribution + noisy peek")
    model_noisy = build_complete_model(include_noisy_actions=True, noise_accuracy=0.8)
    noisy_peek_coin1 = ProbabilisticAction(
        name="noisy_peek_coin1",
        stochastic_rel=lambda s: {
            (s[0], s[1]): 0.8,
            (1 - s[0], s[1]): 0.2,
        },
        alt={"Amy": {"noisy_peek_coin1"}, "Bob": {"noisy_peek_coin1"}}
    )
    model_noisy.actions["noisy_peek_coin1"] = noisy_peek_coin1    
    T3 = export_guarded_string_automaton(
        dist, 
        ["noisy_peek_coin1"], 
        model_noisy, 
        n_coins=2
    )
    print(f"   States: {T3.number_of_states()}, Arcs: {T3.number_of_arcs()}")
    view_fst_clean(T3, "gs_3_with_noisy_peek_coin1", ".")
    
    # Example 4: With det flip_coin1
    print("\n4. Initial distribution + det flip_coin1")
    model_noisy = build_complete_model(include_noisy_actions=True, noise_accuracy=0.8)
    det_flip_coin1 = deterministic_action(
        lambda s: (1 - s[0], s[1]),
        lambda s: True,
        "det_flip_coin1"
    )
    model_noisy.actions["det_flip_coin1"] = det_flip_coin1
    T4 = export_guarded_string_automaton(
        dist, 
        ["det_flip_coin1"], 
        model_noisy, 
        n_coins=2
    )
    print(f"   States: {T4.number_of_states()}, Arcs: {T4.number_of_arcs()}")
    view_fst_clean(T4, "gs_4_with_det_flip_coin1", ".")

    # Example 5: with noisy flip_coin1
    print("\n5. Initial distribution + noisy flip_coin1")
    model_noisy = build_complete_model(include_noisy_actions=True, noise_accuracy=0.8)
    def create_stochastic_flip(coin_index=0, flip_prob=0.5):
        def stochastic_rel(s):
            current_val = s[coin_index]
            flipped_val = 1 - current_val
            
            if coin_index == 0:
                flipped_state = (flipped_val, s[1])
            else:
                flipped_state = (s[0], flipped_val)
            
            return {
                flipped_state: flip_prob,    # 50% flip
                s: 1 - flip_prob             # 50% stay same
            }
        
        return ProbabilisticAction(
            name="stochastic_flip",
            stochastic_rel=stochastic_rel,
            alt={"Amy": {"stochastic_flip"}, "Bob": {"stochastic_flip"}}
        )
    noisy_flip_coin1 = create_stochastic_flip(coin_index=0, flip_prob=0.5)
    model_noisy.actions["noisy_flip_coin1"] = noisy_flip_coin1
    T5 = export_guarded_string_automaton(
        dist, 
        ["noisy_flip_coin1"], 
        model_noisy, 
        n_coins=2
    )
    print(f"   States: {T5.number_of_states()}, Arcs: {T5.number_of_arcs()}")
    view_fst_clean(T5, "gs_5_with_noisy_flip_coin1", ".")
    os.chdir('..')
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("\nGenerated guarded string automata in guarded_strings/:")
    print("  • gs_1_initial_only.png - Branches for initial states")
    print("  • gs_2_with_announce.png - With deterministic action")
    print("  • gs_3_with_noisy_peek.png - With probabilistic action")
    print("\nKey properties:")
    print("  ✓ Each PATH through FST = one possible world")
    print("  ✓ States in FST = positions in sequence")
    print("  ✓ Transitions = bits (0,1) and actions")
    print("  ✓ Weights = probabilities")
    print("\nThis matches Campbell & Rooth's guarded string model!")


if __name__ == "__main__":
    demo_guarded_strings()