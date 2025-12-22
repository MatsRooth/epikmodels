#!/usr/bin/env python3
"""
FST Visualization Script - View Probabilistic EpiKAT FSTs as graphs

This script creates and visualizes FST structures from the probabilistic model.
Outputs are saved as PNG/PDF files that can be viewed in any image viewer.

Requirements:
- hfst_dev (or hfst)
- graphviz

Usage:
    python visualize_fsts.py
"""

import sys
import os

# Try both hfst packages
try:
    import hfst_dev as hfst
    print("✓ Using hfst_dev")
except ImportError:
    try:
        import hfst
        print("✓ Using hfst")
    except ImportError:
        print("ERROR: Neither hfst_dev nor hfst is available")
        sys.exit(1)

try:
    import graphviz
    print("✓ Graphviz available")
except ImportError:
    print("ERROR: graphviz package not available")
    print("Install with: pip install graphviz")
    sys.exit(1)

from probabilistic_epikat import *


def save_fst_visualization(transducer, filename, title="FST", format='png'):
    """
    Save FST visualization to file using AT&T format export.
    
    Args:
        transducer: HFST transducer
        filename: Output filename (without extension)
        title: Title for the graph
        format: Output format ('png', 'pdf', 'svg')
    """
    # Export to AT&T format (text representation)
    att_file = f"{filename}.att"
    
    # write_att expects a file object, not a string
    with open(att_file, 'w') as f:
        transducer.write_att(f)
    
    # Read AT&T format
    with open(att_file, 'r') as f:
        att_lines = f.readlines()
    
    # Parse AT&T format and build graphviz graph
    dot = graphviz.Digraph(comment=title)
    dot.attr(rankdir='LR')
    dot.attr('node', shape='circle')
    
    # Track states and final states
    states = set()
    final_states = set()
    
    for line in att_lines:
        parts = line.strip().split('\t')
        if len(parts) == 1:
            # Final state
            final_states.add(parts[0])
            states.add(parts[0])
        elif len(parts) >= 3:
            # Transition: src dst input output [weight]
            src, dst, input_sym, output_sym = parts[0], parts[1], parts[2], parts[3]
            weight = parts[4] if len(parts) > 4 else None
            
            states.add(src)
            states.add(dst)
            
            # Create label
            if input_sym == output_sym:
                label = input_sym
            else:
                label = f"{input_sym}:{output_sym}"
            
            # Add weight if present
            if weight and float(weight) != 0.0:
                prob = weight_to_prob(float(weight))
                label += f"\n[{prob:.3f}]"
            
            dot.edge(src, dst, label=label)
    
    # Style nodes
    for state in states:
        if state in final_states:
            dot.node(state, state, shape='doublecircle')
        else:
            dot.node(state, state)
    
    # Mark initial state
    dot.node('', '', shape='none')
    dot.edge('', '0', style='bold')
    
    # Render
    output_path = f"{filename}"
    dot.render(output_path, format=format, cleanup=True)
    
    # Clean up AT&T file
    os.remove(att_file)
    
    print(f"  ✓ Saved: {output_path}.{format}")
    return f"{output_path}.{format}"


def visualize_state_distribution():
    """Visualize state distribution FST."""
    print("\n" + "="*70)
    print("VISUALIZATION 1: STATE DISTRIBUTION FST")
    print("="*70)
    
    # Create simple biased distribution
    dist = StateDistribution({
        (1, 0): 0.7,
        (0, 1): 0.3,
    })
    
    print("\nState distribution:")
    print(dist)
    
    # Export FST (without self-loops for cleaner visualization)
    T = export_state_distribution_to_hfst(dist, add_self_loops=False)
    
    print(f"\nFST structure:")
    print(f"  States: {T.number_of_states()}")
    print(f"  Arcs: {T.number_of_arcs()}")
    
    # Visualize
    save_fst_visualization(T, "fst_1_state_distribution", 
                          title="State Distribution FST")


def visualize_single_action():
    """Visualize single action FST."""
    print("\n" + "="*70)
    print("VISUALIZATION 2: SINGLE ACTION FST")
    print("="*70)
    
    # Create a simple deterministic action
    action = deterministic_action(
        lambda s: (1 - s[0], s[1]),
        lambda s: True,
        "flip_coin1"
    )
    
    print("\nAction: flip_coin1 (deterministic)")
    print("Effect: Observe coin1 flipped to the other side")
    
    # Export FST
    T = export_action_to_hfst(action)
    
    print(f"\nFST structure:")
    print(f"  States: {T.number_of_states()}")
    print(f"  Arcs: {T.number_of_arcs()}")
    
    # Visualize
    save_fst_visualization(T, "fst_2_deterministic_action",
                          title="Deterministic Action FST")


def visualize_noisy_action():
    """Visualize noisy action FST."""
    print("\n" + "="*70)
    print("VISUALIZATION 3: NOISY ACTION FST")
    print("="*70)
    
    # Create noisy observation
    actions = create_noisy_peeks(accuracy=0.8)
    noisy_peek = actions["peek_Amy_coin1_H_noisy"]
    
    print("\nAction: peek_Amy_coin1_H_noisy (80% accurate)")
    print("Effect: Observe coin1=H with 80% accuracy, 20% error")
    
    # Export FST
    T = export_action_to_hfst(noisy_peek)
    
    print(f"\nFST structure:")
    print(f"  States: {T.number_of_states()}")
    print(f"  Arcs: {T.number_of_arcs()}")
    
    # Visualize
    save_fst_visualization(T, "fst_3_noisy_action",
                          title="Noisy Action FST (80% accuracy)")


def visualize_composed_sequence():
    """Visualize composed action sequence."""
    print("\n" + "="*70)
    print("VISUALIZATION 4: COMPOSED ACTION SEQUENCE")
    print("="*70)
    
    # Build model
    model = build_complete_model(include_noisy_actions=True, noise_accuracy=0.9)
    
    # Simple sequence: initial state → action
    print("\nSequence: uniform_distribution → peek_Amy_coin1_H_noisy")
    
    # Export complete model
    T = export_complete_model_to_hfst(
        model,
        action_sequence=["peek_Amy_coin1_H_noisy"],
        include_beliefs=False
    )
    
    print(f"\nFST structure:")
    print(f"  States: {T.number_of_states()}")
    print(f"  Arcs: {T.number_of_arcs()}")
    
    # Visualize
    save_fst_visualization(T, "fst_4_composed_sequence",
                          title="Composed: State → Action")


def visualize_minimal_example():
    """Visualize minimal example like in your screenshot."""
    print("\n" + "="*70)
    print("VISUALIZATION 5: MINIMAL EXAMPLE (like your screenshot)")
    print("="*70)
    
    # Create simple regex FST like your example
    print("\nCreating simple FST: a b+")
    T = hfst.regex('a b+')
    
    print(f"\nFST structure:")
    print(f"  States: {T.number_of_states()}")
    print(f"  Arcs: {T.number_of_arcs()}")
    
    print("\nAT&T format:")
    print(T)
    
    # Visualize
    save_fst_visualization(T, "fst_5_simple_regex",
                          title="Simple FST: a b+")


def visualize_intersected():
    """Visualize intersection like in your screenshot."""
    print("\n" + "="*70)
    print("VISUALIZATION 6: INTERSECTION (like your screenshot)")
    print("="*70)
    
    print("\nCreating: (a b+) ∩ (a+ b)")
    x1 = hfst.regex('a b+')
    x2 = hfst.regex('a+ b')
    x1.intersect(x2)
    
    print(f"\nResult FST:")
    print(f"  States: {x1.number_of_states()}")
    print(f"  Arcs: {x1.number_of_arcs()}")
    
    print("\nAT&T format:")
    print(x1)
    
    # Visualize
    save_fst_visualization(x1, "fst_6_intersection",
                          title="Intersection: (a b+) ∩ (a+ b)")


def visualize_weighted_example():
    """Visualize weighted FST with probabilities."""
    print("\n" + "="*70)
    print("VISUALIZATION 7: WEIGHTED FST WITH PROBABILITIES")
    print("="*70)
    
    # Create weighted FST manually
    print("\nCreating weighted FST with probabilities...")
    
    # Build basic transducer
    tr = hfst.HfstBasicTransducer()
    tr.add_state(0)
    tr.add_state(1)
    tr.add_state(2)
    
    # Add transitions with weights (tropical semiring: weight = -log(prob))
    tr.add_transition(0, 1, "observe_H", "observe_H", prob_to_weight(0.9))
    tr.add_transition(0, 1, "observe_T", "observe_T", prob_to_weight(0.1))
    tr.add_transition(1, 2, "confirm", "confirm", prob_to_weight(0.8))
    
    tr.set_final_weight(2, 0.0)
    
    T = hfst.HfstTransducer(tr)
    
    print(f"\nFST structure:")
    print(f"  States: {T.number_of_states()}")
    print(f"  Arcs: {T.number_of_arcs()}")
    
    # Visualize
    save_fst_visualization(T, "fst_7_weighted_probabilities",
                          title="Weighted FST with Probabilities")


def main():
    """Run all visualizations."""
    print("\n" + "="*70)
    print("FST VISUALIZATION SCRIPT".center(70))
    print("="*70)
    print("\nThis script will create FST visualizations and save them as PNG files.")
    print("You can open the PNG files in any image viewer.")
    
    input("\nPress Enter to start...")
    
    # Create output directory
    if not os.path.exists('fst_visualizations'):
        os.makedirs('fst_visualizations')
        print("\n✓ Created directory: fst_visualizations/")
    
    os.chdir('fst_visualizations')
    
    # Run visualizations
    try:
        visualize_minimal_example()
        input("\nPress Enter for next visualization...")
        
        visualize_intersected()
        input("\nPress Enter for next visualization...")
        
        visualize_weighted_example()
        input("\nPress Enter for next visualization...")
        
        visualize_state_distribution()
        input("\nPress Enter for next visualization...")
        
        visualize_single_action()
        input("\nPress Enter for next visualization...")
        
        visualize_noisy_action()
        input("\nPress Enter for next visualization...")
        
        visualize_composed_sequence()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        os.chdir('..')
    
    # Summary
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated visualizations in fst_visualizations/:")
    
    files = [
        "fst_5_simple_regex.png",
        "fst_6_intersection.png", 
        "fst_7_weighted_probabilities.png",
        "fst_1_state_distribution.png",
        "fst_2_deterministic_action.png",
        "fst_3_noisy_action.png",
        "fst_4_composed_sequence.png",
    ]
    
    for f in files:
        if os.path.exists(f"fst_visualizations/{f}"):
            print(f"  ✓ {f}")
    
    print("\nYou can now:")
    print("  • Open these PNG files in any image viewer")
    print("  • Include them in papers/presentations")
    print("  • Compare FST structures visually")
    
    print("\nTo view in Python (without Jupyter):")
    print("  from PIL import Image")
    print("  img = Image.open('fst_visualizations/fst_1_state_distribution.png')")
    print("  img.show()")


if __name__ == "__main__":
    main()