
# HFST Weighted Acceptor Comparison — Conversation Notes

## Problem
Given two weighted acceptors **A** and **B**, compute an **unweighted acceptor** for the strings where

    w_A(x) < w_B(x)

(where lower weight means higher probability).

The work context is **HFST**, but the construction may involve exporting the automata to a simpler graph representation.

---

# Key Idea: Product / Pair Construction

Build a synchronized **product automaton** whose states are pairs:

    (q_A, q_B)

Transitions follow the same input symbol in both automata.

If the arc weights are:

    wA(q_A → q_A')
    wB(q_B → q_B')

then store the **difference weight**

    d += wA − wB

so the product machine accumulates

    Δ(x) = w_A(x) − w_B(x)

A string should be accepted iff

    Δ(x) < 0

---

# Important Limitation

The language

    { x : w_A(x) < w_B(x) }

is **not always regular**.

Example:

If

    w_A(x) = #a(x)
    w_B(x) = #b(x)

then the desired language is

    #a(x) < #b(x)

which is not regular.

Therefore a finite unweighted acceptor **does not always exist**.

---

# Practical Strategy: Product + Pruning

Even though the worst case is non‑regular, many practical machines behave well.

Strategy:

1. Build the synchronized product.
2. Track the **difference value d**.
3. Use **future bounds** to prune states.

At a product state (q_A,q_B), compute bounds on possible future differences:

    L(q_A,q_B) ≤ future difference ≤ U(q_A,q_B)

If

    d + U < 0

then the string will always satisfy the inequality → accept.

If

    d + L ≥ 0

then it can never satisfy the inequality → reject.

Only explore states where

    d + L < 0 ≤ d + U

---

# When This Works Well

Pruning works well when:

• weights come from bounded penalties  
• automata restrict future continuations  
• difference values remain within a finite band

Failure occurs when both positive and negative difference cycles exist that can be repeated arbitrarily.

---

# Extracting Graph Structure from HFST

Instead of exporting to AT&T text, it is easier to convert to a basic graph object.

HFST provides:

    HfstBasicTransducer

which exposes states and transitions.

Conversion syntax:

```python
import hfst

basic = hfst.HfstBasicTransducer(tr)
```

Iterating transitions:

```python
for state in basic.states():
    if basic.is_final_state(state):
        print(state, basic.get_final_weight(state))

    for t in basic.transitions(state):
        print(
            state,
            "->",
            t.get_target_state(),
            t.get_input_symbol(),
            t.get_output_symbol(),
            t.get_weight()
        )
```

---

# Suggested Internal Graph Structure

Example Python structure for algorithms:

```python
from dataclasses import dataclass, field

@dataclass
class Arc:
    target: int
    symbol: str
    weight: float

@dataclass
class WFSA:
    start: int
    finals: dict[int, float] = field(default_factory=dict)
    arcs: dict[int, list[Arc]] = field(default_factory=dict)
```

This makes it easy to:

• construct the product automaton  
• track weight differences  
• implement pruning rules.

---

# Current Status

Confirmed working conversion:

```python
amyBasic = hfst.HfstBasicTransducer(amy)
```

Next steps would be:

1. Export HFST machines to a Python graph structure.
2. Build the synchronized product automaton.
3. Implement difference tracking and pruning.
4. Attempt to collapse the resulting structure into an unweighted acceptor.
