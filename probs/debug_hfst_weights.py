#!/usr/bin/env python3
"""
Debug script to understand HFST weight format in your installation.
Run this to see what type/structure weights have.
"""

import sys
sys.path.insert(0, '.')

from probabilistic_epikat import *

if not HFST_AVAILABLE:
    print("ERROR: HFST not available")
    sys.exit(1)

print("=" * 60)
print("DEBUGGING HFST WEIGHT FORMAT")
print("=" * 60)

# Create simple model
model = build_complete_model(include_noisy_actions=True, noise_accuracy=0.9)

# Export simple transducer
print("\n1. Creating simple transducer...")
T = export_complete_model_to_hfst(
    model,
    action_sequence=["peek_Amy_coin1_H_noisy"],
    include_beliefs=False
)

print(f"   States: {T.number_of_states()}, Arcs: {T.number_of_arcs()}")

# Try to extract paths
print("\n2. Extracting paths with max_cycles=1...")
try:
    paths = T.extract_paths(output='dict', max_cycles=1)
    print(f"   Extracted {len(paths)} paths")
    
    if len(paths) > 0:
        # Examine first few weights
        print("\n3. Examining weight format:")
        for i, (path, weight) in enumerate(list(paths.items())[:3]):
            print(f"\n   Path {i+1}:")
            print(f"     Path string: {path[:50]}...")
            print(f"     Weight type: {type(weight)}")
            print(f"     Weight value: {weight}")
            print(f"     Weight repr: {repr(weight)}")
            
            # Try to extract numeric value
            print(f"     Attempting extraction:")
            extracted = weight
            depth = 0
            while isinstance(extracted, (list, tuple)) and depth < 10:
                print(f"       Depth {depth}: {type(extracted)} with {len(extracted)} elements")
                if len(extracted) > 0:
                    print(f"         Element[0]: {type(extracted[0])} = {extracted[0]}")
                    extracted = extracted[0]
                else:
                    print(f"         Empty container!")
                    break
                depth += 1
            
            print(f"     Final extracted value: {extracted} (type: {type(extracted)})")
            
            # Try conversion
            try:
                numeric = float(extracted)
                print(f"     ✓ Converted to float: {numeric}")
                prob = math.exp(-numeric) if numeric != float('inf') else 0.0
                print(f"     ✓ Probability: {prob:.6f}")
            except Exception as e:
                print(f"     ✗ Conversion failed: {e}")
    
except Exception as e:
    print(f"   ✗ Path extraction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)
print("\nBased on the output above, we need to:")
print("1. Handle the specific weight structure your HFST returns")
print("2. Update extract_paths_safe() accordingly")
print("\nPlease share the output of this script!")