from __future__ import annotations
from dataclasses import dataclass
from typing import Any


# ============================================================
# Base class
# ============================================================

@dataclass(frozen=True)
class EpiExpr:
    """Base class for EpiKAT expressions."""

    # algebraic operators
    def __add__(self, other: "EpiExpr") -> "EpiExpr":
        return Plus(self, other)

    def __mul__(self, other: "EpiExpr") -> "EpiExpr":
        return Dot(self, other)

    def star(self) -> "EpiExpr":
        return Star(self)

    # pretty printing fallback
    def __str__(self):
        return self.__repr__()


# ============================================================
# KAT primitive constructs
# ============================================================

@dataclass(frozen=True)
class Zero(EpiExpr):
    def __repr__(self): return "0"

@dataclass(frozen=True)
class One(EpiExpr):
    def __repr__(self): return "1"

@dataclass(frozen=True)
class Test(EpiExpr):
    """A test: corresponds to a set of states (subidentities)."""
    name: str
    def __repr__(self): return f"[{self.name}]"

@dataclass(frozen=True)
class Action(EpiExpr):
    """Primitive action event."""
    name: str
    def __repr__(self): return self.name


# ============================================================
# KAT operators
# ============================================================

@dataclass(frozen=True)
class Plus(EpiExpr):
    left: EpiExpr
    right: EpiExpr
    def __repr__(self): return f"({self.left} + {self.right})"

@dataclass(frozen=True)
class Dot(EpiExpr):
    left: EpiExpr
    right: EpiExpr
    def __repr__(self): return f"({self.left}·{self.right})"

@dataclass(frozen=True)
class Star(EpiExpr):
    expr: EpiExpr
    def __repr__(self): return f"({self.expr})*"


# ============================================================
# Epistemic operators (Box / Dia)
# ============================================================

@dataclass(frozen=True)
class Box(EpiExpr):
    """□_agent φ : agent a knows (or verifies) φ."""
    agent: str
    body: EpiExpr
    def __repr__(self):
        return f"□_{self.agent}({self.body})"


@dataclass(frozen=True)
class Dia(EpiExpr):
    """◇_agent φ : agent a considers φ epistemically possible."""
    agent: str
    body: EpiExpr
    def __repr__(self):
        return f"◇_{self.agent}({self.body})"
