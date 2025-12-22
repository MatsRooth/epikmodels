#!/usr/bin/env python3
"""
Fixed Noisy Peek Action - Works from ANY state

Creates observation actions that properly model noisy sensors:
- Can be applied from ANY state (no precondition)
- Produces different outputs based on what's observed
- Observation encoded in output symbol
"""

from probabilistic_epikat import *


def create_proper_noisy_peek(
    coin_index: int,
    agent: str = "Amy",
    accuracy: float = 0.9
) -> ProbabilisticAction:
    """
    Create a noisy peek action that works from ANY state.
    
    Key difference from current implementation:
    - No precondition! Can peek from any state.
    - Returns different "observed states" to encode what was seen
    - The world doesn't change, but the symbol shows the observation
    
    Output encoding:
    - If observes H: state → (state, "obs_H")
    - If observes T: state → (state, "obs_T")
    
    For belief updates, we check the observation symbol.
    """
    name = f"peek_{agent}_coin{coin_index+1}"
    
    def stochastic_rel(s: State) -> Dict[State, float]:
        """
        From any state, can observe either H or T.
        Probability depends on true value and sensor accuracy.
        """
        true_value = s[coin_index]
        
        # Return: {state: probability}
        # The state encodes both the world AND the observation
        # But since our states are just (coin1, coin2), we can't encode observation
        
        # WORKAROUND: Return the state, and track observation separately
        # The observation is determined by whether we stay in same state
        
        # If we observe correctly: high probability, stay in state
        # If we observe incorrectly: low probability, "stay in state but with wrong observation"
        
        # This doesn't work either because we need to distinguish the two outcomes!
        
        # REAL SOLUTION: We need to use the ACTION SYMBOLS differently
        # Instead of peek_H vs peek_T (preconditioned)
        # We need: peek (any state) → {peek:H, peek:T} with different probabilities
        
        if true_value == 1:  # Coin is actually H
            # Observe H with prob=accuracy, observe T with prob=1-accuracy
            # Both "stay in state" but we need different symbols
            # Use tuple (state, observation_symbol) but keep only state for FST
            return {
                s: accuracy,  # This represents "observed H correctly"
                # But we also need a branch for "observed T incorrectly"!
            }
        else:  # Coin is actually T  
            return {
                s: 1 - accuracy  # This represents "observed H incorrectly"
            }
    
    return ProbabilisticAction(
        name=name,
        stochastic_rel=stochastic_rel,
        alt={agent: {name}}
    )


# The real solution: we need to change the STATE SPACE!

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


print(__doc__)
print("\nDemonstrating the proper noisy peek:")

# Create proper noisy peek
peek = create_noisy_peek_with_obs_states(coin_index=0, agent="Amy", accuracy=0.9)

print("\n" + "="*70)
print("From state (1, 0) - coin1=H:")
print("="*70)
result = peek.stochastic_rel((1, 0))
for state, prob in result.items():
    print(f"  -> {state} with P={prob:.2f}")

print("\n" + "="*70)
print("From state (0, 1) - coin1=T:")
print("="*70)
result = peek.stochastic_rel((0, 1))
for state, prob in result.items():
    print(f"  -> {state} with P={prob:.2f}")

print("\n" + "="*70)
print("KEY INSIGHT")
print("="*70)
print("""
Now we have TWO branches from each state:
- One for "observed H" (encoded in state as (coins..., "H"))
- One for "observed T" (encoded in state as (coins..., "T"))

The belief update can check the last component of the state
to determine what was observed!

This is the CORRECT way to model noisy observations.
""")