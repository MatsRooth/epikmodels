#!/usr/bin/env python3
"""
Clean FST Viewer - Strips long state strings from labels

This viewer cleans up FST visualizations by:
1. Removing state string prefixes from action labels
2. Showing just action names and probabilities  
3. Using bit sequences for state transitions

Usage:
    from clean_fst_viewer import view_fst_clean
    view_fst_clean(transducer, "my_fst")
"""

import os
import tempfile
import re

try:
    import hfst_dev as hfst
except ImportError:
    import hfst

import graphviz
from probabilistic_epikat import *

def clean_label(label):
    """
    Clean up a transition label by removing state prefixes.
    
    Examples:
        "0" -> "0" (single bit - keep it!)
        "1" -> "1" (single bit - keep it!)
        "obs_H" -> "→H" (observation symbol)
        "obs_T" -> "→T" (observation symbol)
        "peek_Amy_coin1_H_noisy" -> "peek_Amy_c1_H"
        "announceH1" -> "announceH1"
        "1010:1010" -> "ε" (identity/self-loop only)
    """
    # Remove epsilon symbols
    if label == '@_EPSILON_SYMBOL_@':
        return 'ε'
    
    # Handle observation labels
    if label.startswith('obs_'):
        obs_value = label[4:]  # Extract H or T
        return f"→{obs_value}"  # Use arrow to show it's an observation
    
    # First, strip any input:output notation
    if ':' in label:
        parts = label.split(':')
        if len(parts) == 2 and parts[0] == parts[1]:
            # Identity transition
            label = parts[0]
            # Only mark as epsilon if it's a multi-bit self-loop
            if len(label) > 1 and re.match(r'^[01]+$', label):
                return 'ε'
        else:
            label = parts[0]  # Use input side
    
    # IMPORTANT: If it's a SINGLE bit (0 or 1), keep it!
    if label in ['0', '1']:
        return label
    
    # If it's ONLY bits but more than one, it might be a state encoding
    if re.match(r'^[01]+$', label) and len(label) > 1:
        # This is a multi-bit sequence, likely from old export
        # For guarded strings, we should only have single bits
        # Return as-is for now, but ideally shouldn't happen
        return label
    
    # Clean up action names while preserving important details
    if 'peek_Amy_coin' in label or 'peek_Bob_coin' in label:
        # Extract agent, coin number, and observed value
        # "peek_Amy_coin1_H_noisy" -> "peek_Amy_c1_H"
        agent = "Amy" if "Amy" in label else "Bob"
        coin_num = "1" if "coin1" in label else "2"
        observed = "H" if "_H" in label else "T"
        is_noisy = "noisy" in label
        
        if is_noisy:
            return f"peek_{agent}_c{coin_num}_{observed}"
        else:
            return f"peek_{agent}_c{coin_num}_{observed}"
    
    # Handle generic peek labels (without H/T suffix)
    if label.startswith('peek_Amy_') or label.startswith('peek_Bob_'):
        # "peek_Amy_coin1" -> "peek_Amy_c1"
        agent = "Amy" if "Amy" in label else "Bob"
        coin_num = "1" if "coin1" in label else "2"
        return f"peek_{agent}_c{coin_num}"
    
    if 'announce' in label:
        # "announceH1" or "announceH2" - preserve exactly
        match = re.search(r'announce[HT][12]', label)
        if match:
            return match.group(0)
        return 'announce'
    
    if 'turn' in label or 'flip' in label:
        # "turn_coin1" or "flip_coin1" - preserve coin number
        match = re.search(r'(turn|flip)_coin[12]', label)
        if match:
            return match.group(0)
        if 'turn' in label:
            return 'turn'
        if 'flip' in label:
            return 'flip'
    
    # If it contains action names but also has state bits mixed in,
    # try to extract just the action part
    if any(action in label for action in ['peek', 'announce', 'turn', 'flip']):
        # Remove leading/trailing bit sequences
        label = re.sub(r'^[01]{2,}', '', label)
        label = re.sub(r'[01]{2,}$', '', label)
        
        # Re-apply action cleaning
        if 'peek' in label:
            agent = "Amy" if "Amy" in label else "Bob" if "Bob" in label else ""
            coin_num = "1" if "coin1" in label or "c1" in label else "2"
            obs = "H" if "H" in label else "T" if "T" in label else ""
            if agent and obs:
                return f"peek_{agent}_c{coin_num}_{obs}"
            if agent:
                return f"peek_{agent}_c{coin_num}"
            return 'peek'
        if 'announce' in label:
            return 'announce'
        if 'turn' in label:
            return 'turn'
        if 'flip' in label:
            return 'flip'
    
    # If we get here and it's empty, return epsilon
    if not label or label.isspace():
        return 'ε'
    
    return label


def view_fst_clean(transducer, name="fst", output_dir=".", format='png', view_immediately=False):
    """
    Visualize an HFST transducer with CLEAN labels.
    
    This version strips state encodings from labels to show only
    meaningful action names and probabilities.
    
    Args:
        transducer: HFST transducer to visualize
        name: Base name for output file
        output_dir: Directory to save output
        format: Output format ('png', 'pdf', 'svg')
        view_immediately: If True, try to open the image
    
    Returns:
        Path to generated image file
    """
    # Create temp file for AT&T format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.att', delete=False) as f:
        att_file = f.name
    
    try:
        # Export to AT&T format - write_att expects file object
        with open(att_file, 'w') as f:
            transducer.write_att(f)
        
        # Read AT&T format
        with open(att_file, 'r') as f:
            att_lines = f.readlines()
        
        # Parse and build graph
        dot = graphviz.Digraph(comment=name)
        dot.attr(rankdir='LR')
        dot.attr('node', shape='circle', fontsize='14', fontname='Arial')
        dot.attr('edge', fontsize='12', fontname='Arial')
        
        states = set()
        final_states = {}
        
        for line in att_lines:
            parts = line.strip().split('\t')
            
            if len(parts) == 1:
                # Final state (no weight)
                final_states[parts[0]] = 0.0
                states.add(parts[0])
            elif len(parts) == 2:
                # Final state with weight
                final_states[parts[0]] = float(parts[1])
                states.add(parts[0])
            elif len(parts) >= 3:
                # Transition
                src, dst = parts[0], parts[1]
                input_sym = parts[2] if len(parts) > 2 else '@_EPSILON_SYMBOL_@'
                output_sym = parts[3] if len(parts) > 3 else input_sym
                weight = float(parts[4]) if len(parts) > 4 else 0.0
                
                states.add(src)
                states.add(dst)
                
                # Create cleaned label
                if input_sym == output_sym:
                    raw_label = input_sym
                else:
                    raw_label = f"{input_sym}:{output_sym}"
                
                # CLEAN THE LABEL
                label = clean_label(raw_label)
                
                # Add probability if non-zero weight
                if abs(weight) > 0.001:
                    try:
                        import math
                        prob = math.exp(-weight)
                        label += f"\\n[{prob:.3f}]"
                    except:
                        label += f"\\n[w={weight:.3f}]"
                
                # Skip epsilon self-loops for cleaner visualization
                if label == 'ε' and src == dst:
                    continue
                
                dot.edge(src, dst, label=label)
        
        # Style nodes - use q0, q1, q2 naming like Campbell & Rooth
        state_list = sorted(states, key=lambda x: int(x))
        for i, state in enumerate(state_list):
            node_label = f"q{i}"
            
            if state in final_states:
                weight = final_states[state]
                if abs(weight) > 0.001:
                    dot.node(state, f"{node_label}\\n[{weight:.2f}]", shape='doublecircle')
                else:
                    dot.node(state, node_label, shape='doublecircle')
            else:
                dot.node(state, node_label)
        
        # Mark initial state with invisible node and arrow
        dot.node('__start__', '', shape='none', width='0', height='0')
        dot.edge('__start__', '0', style='bold')
        
        # Render
        output_path = os.path.join(output_dir, name)
        dot.render(output_path, format=format, cleanup=True, view=view_immediately)
        
        result_file = f"{output_path}.{format}"
        print(f"✓ Saved FST visualization: {result_file}")
        
        return result_file
        
    finally:
        # Clean up temp file
        if os.path.exists(att_file):
            os.remove(att_file)


def demo():
    """Demo the clean FST viewer."""
    print("="*70)
    print("CLEAN FST VIEWER DEMO".center(70))
    print("="*70)
    
    # Create output directory
    output_dir = "clean_fst_views"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n✓ Output directory: {output_dir}/")
    
    # Example 1: State distribution
    print("\n1. State Distribution (cleaned)")
    dist = StateDistribution.from_dict({
        (1, 0): 0.7,
        (0, 1): 0.3,
    })
    T1 = export_state_distribution_to_hfst(dist, add_self_loops=False)
    view_fst_clean(T1, "clean_state_dist", output_dir)
    
    # Example 2: Noisy action
    print("\n2. Noisy Action (cleaned)")
    actions = create_noisy_peeks(coin_index=0, accuracy=0.8)
    T2 = export_action_to_hfst(actions["peek_Amy_coin1_H_noisy"])
    view_fst_clean(T2, "clean_noisy_action", output_dir)
    
    # Example 3: Complete sequence
    print("\n3. Complete Sequence (cleaned)")
    model = build_complete_model(include_noisy_actions=True, noise_accuracy=0.9)
    T3 = export_complete_model_to_hfst(
        model,
        action_sequence=["peek_Amy_coin1_H_noisy"],
        include_beliefs=False
    )
    view_fst_clean(T3, "clean_sequence", output_dir)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print(f"\nCheck {output_dir}/ for cleaned visualizations!")
    print("\nCompare with original visualizations to see the difference:")
    print("  • State strings removed from labels")
    print("  • Clean action names (peek_H, announce, etc.)")
    print("  • Node labels like q0, q1, q2 (Campbell & Rooth style)")
    print("  • Epsilon self-loops hidden")


if __name__ == "__main__":
    demo()