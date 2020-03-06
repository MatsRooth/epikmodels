
# WH operator forming a different-answer relation, the complement of the
# partition-semantics denotation.
define WH(Y,X) [[X & Y] .x. [X - Y]] | [[X - Y] .x. [X & Y]];


# Know Q. Subtract away where we can get using R to cells that are
# different according to Q. Q should be the complement of the
# partition-semantics equivalence relation.
define Kw(Q,R,X) X - [[R.i .o. Q .o. R] & X];

# Agent R knows fluent F in space of worlds X.
define Know(R,F,X) Kw(WH(Cn(X,F),X),R,X);

