# epikat_two_coin_model.py
#
# Two-coin model for Epistemic KAT (Campbell & Rooth).
#
# Terminology:
#   - state = a single bit-vector (coin1, coin2)
#   - subidentity = a set of states (public information)
#
# Actions are relations on states:
#       rel : State -> Set[State]
#
# Worlds (guarded strings) will be sequences of (subidentity, action-label, ...)

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Set, Tuple


# ============================================================
# 1. States and Subidentities
# ============================================================

State = Tuple[int, int]     # (coin1, coin2), 0=T, 1=H

ALL_STATES: FrozenSet[State] = frozenset({
    (0,0), (0,1), (1,0), (1,1)
})


@dataclass(frozen=True)
class Subidentity:
    """A public-information cell: a set of possible underlying states."""
    states: FrozenSet[State]

    @staticmethod
    def full() -> "Subidentity":
        return Subidentity(ALL_STATES)

    def restrict(self, predicate: Callable[[State], bool]) -> "Subidentity":
        return Subidentity(frozenset(s for s in self.states if predicate(s)))

    def __repr__(self):
        items = ",".join("".join(str(x) for x in st) for st in sorted(self.states))
        return f"{{{items}}}"


# ============================================================
# 2. Epistemic Relations on States
# ============================================================

@dataclass(frozen=True)
class EpistemicRelation:
    """An agent's equivalence relation on underlying states."""
    name: str
    classes: FrozenSet[FrozenSet[State]]


def partition_by(key_fn: Callable[[State], int]) -> FrozenSet[FrozenSet[State]]:
    buckets: Dict[int, Set[State]] = {}
    for s in ALL_STATES:
        k = key_fn(s)
        buckets.setdefault(k, set()).add(s)
    return frozenset(frozenset(b) for b in buckets.values())


# Amy sees coin1 exactly
Amy_rel = EpistemicRelation(
    "Amy",
    classes=partition_by(lambda st: st[0])
)

# Bob sees coin2 exactly
Bob_rel = EpistemicRelation(
    "Bob",
    classes=partition_by(lambda st: st[1])
)


# ============================================================
# 3. Action Semantics  (State → Set[State])
# ============================================================

@dataclass(frozen=True)
class ActionSemantics:
    """
    Semantics of an event/action.
    rel(s): set of successor states when this action executes at s.
    alt[agent]: set of labels agent may confuse this action with.
    """
    name: str
    rel: Callable[[State], Set[State]]
    alt: Dict[str, Set[str]]


# --- Helper constructors -----------------------------------------------------

def deterministic(update_fn: Callable[[State], State],
                  precond_fn: Callable[[State], bool] = lambda s: True):
    """
    Deterministic public action:
       if precond_fn(s): {update_fn(s)} else: ∅
    """
    def rel(s: State) -> Set[State]:
        if precond_fn(s):
            return {update_fn(s)}
        return set()
    return rel


def identity(precond_fn: Callable[[State], bool] = lambda s: True):
    """
    Identity on states (private sensing action):
       if precond_fn(s): {s} else: ∅
    """
    def rel(s: State) -> Set[State]:
        if precond_fn(s):
            return {s}
        return set()
    return rel


# ============================================================
# 4. Concrete Actions
# ============================================================

# ---------- Announcement preconditions ----------
def pre_coin1_H(s: State) -> bool: return s[0] == 1
def pre_coin1_T(s: State) -> bool: return s[0] == 0
def pre_coin2_H(s: State) -> bool: return s[1] == 1
def pre_coin2_T(s: State) -> bool: return s[1] == 0

# ---------- Announcements ----------
announce_coin1_H = ActionSemantics(
    name="announce_coin1_H",
    rel=identity(pre_coin1_H),
    alt={"Amy": {"announce_coin1_H"}, "Bob": {"announce_coin1_H"}}
)

announce_coin1_T = ActionSemantics(
    name="announce_coin1_T",
    rel=identity(pre_coin1_T),
    alt={"Amy": {"announce_coin1_T"}, "Bob": {"announce_coin1_T"}}
)

announce_coin2_H = ActionSemantics(
    name="announce_coin2_H",
    rel=identity(pre_coin2_H),
    alt={"Amy": {"announce_coin2_H"}, "Bob": {"announce_coin2_H"}}
)

announce_coin2_T = ActionSemantics(
    name="announce_coin2_T",
    rel=identity(pre_coin2_T),
    alt={"Amy": {"announce_coin2_T"}, "Bob": {"announce_coin2_T"}}
)


# ============================================================
# 4a. Refined peeks for BOTH agents and BOTH coins
# ============================================================

# ---- Preconditions reused for both agents ----
def pre_peek_coin1_H(s: State) -> bool: return s[0] == 1
def pre_peek_coin1_T(s: State) -> bool: return s[0] == 0

def pre_peek_coin2_H(s: State) -> bool: return s[1] == 1
def pre_peek_coin2_T(s: State) -> bool: return s[1] == 0


# ---------- Amy peeks coin1 ----------
peek_Amy_coin1_H = ActionSemantics(
    name="peek_Amy_coin1_H",
    rel=identity(pre_peek_coin1_H),
    alt={
        "Amy": {"peek_Amy_coin1_H"},                          # she knows the outcome
        "Bob": {"peek_Amy_coin1_H", "peek_Amy_coin1_T"}       # Bob cannot tell
    }
)

peek_Amy_coin1_T = ActionSemantics(
    name="peek_Amy_coin1_T",
    rel=identity(pre_peek_coin1_T),
    alt={
        "Amy": {"peek_Amy_coin1_T"},
        "Bob": {"peek_Amy_coin1_H", "peek_Amy_coin1_T"}
    }
)


# ---------- Amy peeks coin2 ----------
peek_Amy_coin2_H = ActionSemantics(
    name="peek_Amy_coin2_H",
    rel=identity(pre_peek_coin2_H),
    alt={
        "Amy": {"peek_Amy_coin2_H"},
        "Bob": {"peek_Amy_coin2_H", "peek_Amy_coin2_T"}
    }
)

peek_Amy_coin2_T = ActionSemantics(
    name="peek_Amy_coin2_T",
    rel=identity(pre_peek_coin2_T),
    alt={
        "Amy": {"peek_Amy_coin2_T"},
        "Bob": {"peek_Amy_coin2_H", "peek_Amy_coin2_T"}
    }
)


# ---------- Bob peeks coin1 ----------
peek_Bob_coin1_H = ActionSemantics(
    name="peek_Bob_coin1_H",
    rel=identity(pre_peek_coin1_H),
    alt={
        "Bob": {"peek_Bob_coin1_H"},
        "Amy": {"peek_Bob_coin1_H", "peek_Bob_coin1_T"}
    }
)

peek_Bob_coin1_T = ActionSemantics(
    name="peek_Bob_coin1_T",
    rel=identity(pre_peek_coin1_T),
    alt={
        "Bob": {"peek_Bob_coin1_T"},
        "Amy": {"peek_Bob_coin1_H", "peek_Bob_coin1_T"}
    }
)


# ---------- Bob peeks coin2 ----------
peek_Bob_coin2_H = ActionSemantics(
    name="peek_Bob_coin2_H",
    rel=identity(pre_peek_coin2_H),
    alt={
        "Bob": {"peek_Bob_coin2_H"},
        "Amy": {"peek_Bob_coin2_H", "peek_Bob_coin2_T"}
    }
)

peek_Bob_coin2_T = ActionSemantics(
    name="peek_Bob_coin2_T",
    rel=identity(pre_peek_coin2_T),
    alt={
        "Bob": {"peek_Bob_coin2_T"},
        "Amy": {"peek_Bob_coin2_H", "peek_Bob_coin2_T"}
    }
)


# ============================================================
# 4b. World-changing action
# ============================================================

def pre_turn_1_HT(s: State) -> bool: return s[0] == 1
def upd_turn_1_HT(s: State) -> State: return (0, s[1])

turn_1_HT = ActionSemantics(
    name="turn_1_HT",
    rel=deterministic(upd_turn_1_HT, pre_turn_1_HT),
    alt={"Amy": {"turn_1_HT"}, "Bob": {"turn_1_HT"}}
)


# ============================================================
# 4c. Optional distinct no-op actions
# ============================================================

Bob_burp = ActionSemantics(
    name="Bob_burp",
    rel=identity(lambda s: True),
    alt={"Amy": {"Bob_burp", "Bob_barf"},
         "Bob": {"Bob_burp"}}
)

Bob_barf = ActionSemantics(
    name="Bob_barf",
    rel=identity(lambda s: True),
    alt={"Amy": {"Bob_burp", "Bob_barf"},
         "Bob": {"Bob_barf"}}
)


# ============================================================
# 5. Model container
# ============================================================

@dataclass
class TwoCoinModel:
    agents: FrozenSet[str]
    epistemic: Dict[str, EpistemicRelation]
    actions: Dict[str, ActionSemantics]


def build_model() -> TwoCoinModel:
    """
    Build the canonical two-coin model including:
      * Amy and Bob's epistemic partitions
      * announcements
      * peeks for BOTH agents on BOTH coins
      * a world-changing turn action
      * optional burp/barf actions
    """
    return TwoCoinModel(
        agents=frozenset({"Amy", "Bob"}),
        epistemic={
            "Amy": Amy_rel,
            "Bob": Bob_rel,
        },
        actions={
            "announce_coin1_H": announce_coin1_H,
            "announce_coin1_T": announce_coin1_T,
            "announce_coin2_H": announce_coin2_H,
            "announce_coin2_T": announce_coin2_T,

            "peek_Amy_coin1_H": peek_Amy_coin1_H,
            "peek_Amy_coin1_T": peek_Amy_coin1_T,
            "peek_Amy_coin2_H": peek_Amy_coin2_H,
            "peek_Amy_coin2_T": peek_Amy_coin2_T,

            "peek_Bob_coin1_H": peek_Bob_coin1_H,
            "peek_Bob_coin1_T": peek_Bob_coin1_T,
            "peek_Bob_coin2_H": peek_Bob_coin2_H,
            "peek_Bob_coin2_T": peek_Bob_coin2_T,

            "turn_1_HT": turn_1_HT,

            "Bob_burp": Bob_burp,
            "Bob_barf": Bob_barf,
        }
    )

