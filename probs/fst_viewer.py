#!/usr/bin/env python3
"""
Simple FST Viewer - Quick visualization utility

This is a simple utility to visualize any HFST transducer and save as an image.
Use this for quick ad-hoc FST visualization without Jupyter.

Usage:
    python fst_viewer.py
    
    # Or import and use in your own scripts:
    from fst_viewer import view_fst
    view_fst(my_transducer, "my_fst_name")
"""

import os
import tempfile

try:
    import hfst_dev as hfst
except ImportError:
    import hfst

import graphviz


def view_fst(transducer, name="fst", output_dir=".", format='png', view_immediately=True):
    """
    Visualize an HFST transducer and save as image.
    
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
        dot.attr('node', shape='circle', fontsize='12')
        dot.attr('edge', fontsize='10')
        
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
                
                # Create label
                if input_sym == output_sym:
                    if input_sym == '@_EPSILON_SYMBOL_@':
                        label = 'ε'
                    else:
                        label = input_sym
                else:
                    label = f"{input_sym}:{output_sym}"
                
                # Add weight if non-zero
                if abs(weight) > 0.001:
                    try:
                        import math
                        prob = math.exp(-weight)
                        label += f"\\n[{prob:.3f}]"
                    except:
                        label += f"\\n[w={weight:.3f}]"
                
                dot.edge(src, dst, label=label)
        
        # Style nodes
        for state in states:
            if state in final_states:
                weight = final_states[state]
                if abs(weight) > 0.001:
                    dot.node(state, f"{state}\\n[{weight:.2f}]", shape='doublecircle')
                else:
                    dot.node(state, state, shape='doublecircle')
            else:
                dot.node(state, state)
        
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
    """Demo the FST viewer with examples."""
    print("="*70)
    print("FST VIEWER DEMO".center(70))
    print("="*70)
    
    # Create output directory
    output_dir = "fst_views"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n✓ Created output directory: {output_dir}/")
    
    # Example 1: Simple regex
    print("\n1. Simple regex: a b+")
    t1 = hfst.regex('a b+')
    view_fst(t1, "example_1_simple", output_dir, view_immediately=False)
    
    # Example 2: Intersection
    print("\n2. Intersection: (a b+) ∩ (a+ b)")
    x1 = hfst.regex('a b+')
    x2 = hfst.regex('a+ b')
    x1.intersect(x2)
    view_fst(x1, "example_2_intersection", output_dir, view_immediately=False)
    
    # Example 3: Union
    print("\n3. Union: (cat) | (dog)")
    t3 = hfst.regex('cat|dog')
    view_fst(t3, "example_3_union", output_dir, view_immediately=False)
    
    # Example 4: Kleene star
    print("\n4. Kleene star: (a b)*")
    t4 = hfst.regex('[a b]*')
    view_fst(t4, "example_4_kleene_star", output_dir, view_immediately=False)
    
    # Example 5: Transducer
    print("\n5. Transducer: a:x b:y")
    t5 = hfst.regex('a:x b:y')
    view_fst(t5, "example_5_transducer", output_dir, view_immediately=False)
    
    print("\n" + "="*70)
    print("Demo complete! Check the fst_views/ directory for images.")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        print("FST Viewer Utility")
        print("\nUsage:")
        print("  python fst_viewer.py demo    # Run demo examples")
        print("\nOr import in your scripts:")
        print("  from fst_viewer import view_fst")
        print("  view_fst(my_transducer, 'my_fst')")
        print("\nRunning demo...")
        demo()