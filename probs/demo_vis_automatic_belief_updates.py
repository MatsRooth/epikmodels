#!/usr/bin/env python3
"""
Epistemic Guarded String Export - Beliefs Integrated into FST

This module creates FSTs that show BOTH:
1. Objective world states and actions (guarded strings)
2. Agent's subjective beliefs at each point (epistemic layer)

The FST structure shows:
- World paths (objective reality)
- Belief annotations on nodes (subjective perspective)
- Belief updates after observations (Bayesian inference)

Example visualization:
    q0 [Amy: uncertain]
     |
     +--(1)--> q1 --(0)--> q2 [Amy: P(H)=0.5]
     |                      |
     |                      +--[peek_H]--> q3 [Amy: P(H)=0.9]
     |
     +--(0)--> q4 --(1)--> q5 [Amy: P(H)=0.5]
                            |
                            +--[peek_H]--> q6 [Amy: P(H)=0.1]
"""
import tempfile
import graphviz
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
from demo_automatic_belief_updates import update_belief_simple
from clean_fst_vis import view_fst_clean, clean_label
from typing import Dict, List, Tuple
import os


def state_to_bits(state):
    """Convert state to bit string."""
    return ''.join(str(b) for b in state)


def export_epistemic_guarded_string(
    initial_dist: StateDistribution,
    initial_beliefs: Dict[str, EpistemicBeliefState],
    action_sequence: List[str],
    model: 'ProbabilisticTwoCoinModel',
    agent_equiv_classes: Dict[str, List[FrozenSet[State]]],
    sensor_accuracies: Dict[str, float],
    n_coins: int = 2
) -> 'hfst.HfstTransducer':
    """
    Export guarded string FST with integrated belief states.
    
    Creates an FST where:
    - Nodes contain belief annotations
    - Paths show world evolution
    - Belief updates happen automatically after observations
    
    Args:
        initial_dist: Initial state distribution
        initial_beliefs: Initial beliefs for each agent
        action_sequence: Sequence of actions
        model: Probabilistic model
        agent_equiv_classes: Equivalence classes for each agent
        sensor_accuracies: Sensor accuracy for each agent's observations
        n_coins: Number of coins
    
    Returns:
        HFST transducer with belief annotations
    """
    if not HFST_AVAILABLE:
        raise ImportError("HFST not available")
    
    tr = hfst.HfstBasicTransducer()
    
    # Track: (world_state, {agent: belief_state}) -> node_id
    # This maps each (world, beliefs) configuration to a node
    state_belief_to_node = {}
    node_counter = 0
    
    # Initial node
    tr.add_state(node_counter)
    initial_node = node_counter
    node_counter += 1
    
    # Track branches: list of (world_state, beliefs_dict, node_id, cumulative_prob)
    branches = []
    
    # Create initial branches for each world state
    for world_state, init_prob in initial_dist.distribution.items():
        # Output bits of initial state
        prev_node = initial_node
        bits = state_to_bits(world_state)
        
        for bit_idx, bit in enumerate(bits):
            next_node = node_counter
            node_counter += 1
            tr.add_state(next_node)
            
            # Weight on last bit
            weight = prob_to_weight(init_prob) if bit_idx == len(bits) - 1 else 0.0
            tr.add_transition(prev_node, next_node, bit, bit, weight)
            prev_node = next_node
        
        # At this node, store initial beliefs
        branches.append((world_state, dict(initial_beliefs), prev_node, init_prob))
    
    # Apply actions sequentially
    for action_idx, action_name in enumerate(action_sequence):
        action = model.actions[action_name]
        new_branches = []
        
        for world_state, beliefs, node, cumulative_prob in branches:
            # Apply action to world state
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
                    
                    # Update beliefs for each agent based on observation
                    # KEY FIX: Determine what was ACTUALLY OBSERVED
                    # by checking if this transition represents correct or incorrect observation
                    updated_beliefs = {}
                    for agent, belief in beliefs.items():
                        if agent in agent_equiv_classes:
                            # Determine what this agent observes
                            # For peek actions, agent observes their coin
                            if f"peek_{agent}" in action_name:
                                # Agent made an observation
                                # Determine which class they OBSERVED (not true state!)
                                equiv_classes = agent_equiv_classes[agent]
                                
                                # The observation is determined by whether this was
                                # a correct observation or an error
                                # If next_state == world_state, observation was correct
                                # Otherwise, observation was incorrect
                                
                                if next_state == world_state:
                                    # CORRECT observation - saw what's actually there
                                    # Find which class world_state is in
                                    for idx, equiv_class in enumerate(equiv_classes):
                                        if world_state in equiv_class:
                                            observed_class_idx = idx
                                            break
                                else:
                                    # ERROR observation - saw the WRONG thing
                                    # Find which class they would have seen (the other one)
                                    for idx, equiv_class in enumerate(equiv_classes):
                                        if world_state not in equiv_class:
                                            observed_class_idx = idx
                                            break
                                
                                # Update belief based on what they observed
                                accuracy = sensor_accuracies.get(agent, 1.0)
                                updated_beliefs[agent] = update_belief_simple(
                                    belief,
                                    list(equiv_classes),
                                    observed_class_idx,
                                    accuracy
                                )
                            else:
                                # No observation, belief unchanged
                                updated_beliefs[agent] = belief
                        else:
                            updated_beliefs[agent] = belief
                    
                    # Add next state bits
                    prev_node = action_node
                    bits = state_to_bits(next_state)
                    for bit in bits:
                        next_node = node_counter
                        node_counter += 1
                        tr.add_state(next_node)
                        tr.add_transition(prev_node, next_node, bit, bit, 0.0)
                        prev_node = next_node
                    
                    # Track new branch with updated beliefs
                    new_cumulative = cumulative_prob * action_prob
                    new_branches.append((next_state, updated_beliefs, prev_node, new_cumulative))
        
        branches = new_branches
    
    # Mark final states
    for world_state, beliefs, node, prob in branches:
        tr.set_final_weight(node, 0.0)
    
    # Store belief annotations as metadata (for visualization)
    # We'll encode this in the transducer for later extraction
    belief_annotations = {}
    for world_state, beliefs, node, prob in branches:
        belief_annotations[node] = beliefs
    
    T = hfst.HfstTransducer(tr)
    
    # Attach metadata (Python-level only, not in FST)
    T._belief_annotations = belief_annotations
    
    return T


def visualize_epistemic_fst(
    transducer: 'hfst.HfstTransducer',
    filename: str,
    show_beliefs: bool = True
):
    """
    Visualize epistemic FST with belief annotations.
    
    This is like view_fst_clean but adds belief information to node labels.
    """
    
    # Export to AT&T format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.att', delete=False) as f:
        att_file = f.name
    
    try:
        with open(att_file, 'w') as f:
            transducer.write_att(f)
        
        with open(att_file, 'r') as f:
            att_lines = f.readlines()
        
        # Parse AT&T format
        dot = graphviz.Digraph(comment=filename)
        dot.attr(rankdir='LR')
        dot.attr('node', shape='circle', fontsize='10', fontname='Arial')
        dot.attr('edge', fontsize='10', fontname='Arial')
        
        states = set()
        final_states = {}
        transitions = []
        
        for line in att_lines:
            parts = line.strip().split('\t')
            
            if len(parts) == 1:
                final_states[parts[0]] = 0.0
                states.add(parts[0])
            elif len(parts) == 2:
                final_states[parts[0]] = float(parts[1])
                states.add(parts[0])
            elif len(parts) >= 3:
                src, dst = parts[0], parts[1]
                input_sym = parts[2]
                output_sym = parts[3] if len(parts) > 3 else input_sym
                weight = float(parts[4]) if len(parts) > 4 else 0.0
                
                states.add(src)
                states.add(dst)
                transitions.append((src, dst, input_sym, output_sym, weight))
        
        # Get belief annotations if available
        belief_annotations = getattr(transducer, '_belief_annotations', {})
        
        # Create nodes with belief labels
        state_list = sorted(states, key=lambda x: int(x))
        for i, state in enumerate(state_list):
            node_label = f"q{i}"
            
            # Add belief information if available
            if show_beliefs and int(state) in belief_annotations:
                beliefs = belief_annotations[int(state)]
                belief_str = ""
                for agent, belief in beliefs.items():
                    # Get most likely class and probability
                    most_likely, prob = belief.most_likely_class()
                    belief_str += f"\\n{agent}:{prob:.2f}"
                node_label += belief_str
            
            if state in final_states:
                dot.node(state, node_label, shape='doublecircle')
            else:
                dot.node(state, node_label)
        
        # Add transitions with cleaned labels
        for src, dst, inp, out, weight in transitions:
            label = clean_label(inp if inp == out else f"{inp}:{out}")
            
            if abs(weight) > 0.001:
                import math
                prob = math.exp(-weight)
                label += f"\\n[{prob:.3f}]"
            
            # Skip epsilon self-loops
            if label == 'ε' and src == dst:
                continue
            
            dot.edge(src, dst, label=label)
        
        # Initial state marker
        dot.node('__start__', '', shape='none', width='0', height='0')
        dot.edge('__start__', '0', style='bold')
        
        # Render
        dot.render(filename, format='png', cleanup=True, view=False)
        print(f"✓ Saved epistemic FST: {filename}.png")
        
    finally:
        if os.path.exists(att_file):
            os.remove(att_file)

def create_state_with_observation(base_state: State, observation: str) -> tuple:
    """
    Augment state with observation.
    New state: (coin1, coin2, last_observation)
    """
    return (*base_state, observation)


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
            last_obs = None
        else:
            coin1, coin2, last_obs = s
        
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

def demo_epistemic_fst():
    """Demonstrate epistemic guarded string FSTs."""
    print("="*70)
    print("EPISTEMIC GUARDED STRING FSTs")
    print("="*70)
    print("\nIntegrating beliefs into FST structure...")
    
    if not os.path.exists('epistemic_fsts'):
        os.makedirs('epistemic_fsts')
    os.chdir('epistemic_fsts')
    
    # Setup
    initial_dist = StateDistribution({
        (1, 0): 0.25,
        (0, 1): 0.25,
        (1, 1): 0.25,
        (0, 0): 0.25
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
    
    # Example 1: Initial state only
    print("\n1. Initial state with beliefs")
    model = build_complete_model(include_noisy_actions=True, noise_accuracy=0.9)
    
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
    peek_Amy_coin1 = create_noisy_peek_with_obs_states(coin_index=0, agent="Amy", accuracy=0.9)
    model.actions["peek_Amy_coin1"] = peek_Amy_coin1  
    T2 = export_epistemic_guarded_string(
        initial_dist,
        {"Amy": amy_initial},
        ["peek_Amy_coin1"],
        model,
        {"Amy": amy_equiv_classes},
        {"Amy": 0.9},
        n_coins=2
    )
    
    print(f"   States: {T2.number_of_states()}, Arcs: {T2.number_of_arcs()}")
    visualize_epistemic_fst(T2, "epistemic_2_amy_one_noisy_peek", show_beliefs=True)
    print("   → Shows AUTOMATIC belief update after observation")
    print("   → Different paths show different belief updates")
    
    # Example 3: Two observations
    print("\n3. After two peeks (compounding belief updates)")
    
    T3 = export_epistemic_guarded_string(
        initial_dist,
        {"Amy": amy_initial},
        ["peek_Amy_coin1", "peek_Amy_coin1"],
        model,
        {"Amy": amy_equiv_classes},
        {"Amy": 0.9},
        n_coins=2
    )
    
    print(f"   States: {T3.number_of_states()}, Arcs: {T3.number_of_arcs()}")
    visualize_epistemic_fst(T3, "epistemic_3_two_peeks", show_beliefs=True)
    print("   → Shows belief compounding through multiple observations")
    
    os.chdir('..')
    
    print("\n" + "="*70)
    print("EPISTEMIC FST VISUALIZATION COMPLETE")
    print("="*70)
    print("\nCreated epistemic FSTs in epistemic_fsts/:")
    print("  • epistemic_1_initial.png - Beliefs on initial branches")
    print("  • epistemic_2_one_peek.png - Automatic belief update after observation")
    print("  • epistemic_3_two_peeks.png - Compounding updates")
    print("\nKey features:")
    print("  ✓ Beliefs shown on FST nodes")
    print("  ✓ Automatic Bayesian updates")
    print("  ✓ Different paths = different belief trajectories")
    print("  ✓ Combines objective (world) and subjective (belief) layers")
    print("\nThis visualizes Campbell & Rooth's epistemic semantics")
    print("with probabilistic beliefs integrated!")


if __name__ == "__main__":
    demo_epistemic_fst()