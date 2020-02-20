define Bool ["0"|"1"];
define St0 [Bool Bool Bool Bool];
define M1 [1 Bool Bool Bool];
define M2 [Bool 1 Bool Bool];
define L1 [Bool Bool 1 Bool];
define L2 [Bool Bool Bool 1];
define St St0 & (((M1 | M2) & (L1 | L2)) & (St0 - (L1 & L2)));
define Nst(X) [St - X];
define M1 St & M1;
define M2 St & M2;
define L1 St & L1;
define L2 St & L2;
define UnequalStPair [M1 Nst(M1)] | [Nst(M1) M1] | [M2 Nst(M2)] | [Nst(M2) M2] | [L1 Nst(L1)] | [Nst(L1) L1] | [L2 Nst(L2)] | [Nst(L2) L2];
define c1%_yes [[[[L2 c1%_yes] [L1 & [St - L2]]] & [[[M1 c1%_yes] M1] | [[[St - M1] c1%_yes] [St - M1]]]] & [[[M2 c1%_yes] M2] | [[[St - M2] c1%_yes] [St - M2]]]];
define c1%_no [[[[L2 c1%_no] [L1 & [St - L2]]] & [[[M1 c1%_no] M1] | [[[St - M1] c1%_no] [St - M1]]]] & [[[M2 c1%_no] M2] | [[[St - M2] c1%_no] [St - M2]]]];
define c2%_yes [[[[L1 c2%_yes] [L2 & [St - L1]]] & [[[M1 c2%_yes] M1] | [[[St - M1] c2%_yes] [St - M1]]]] & [[[M2 c2%_yes] M2] | [[[St - M2] c2%_yes] [St - M2]]]];
define c2%_no [[[[L1 c2%_no] [L2 & [St - L1]]] & [[[M1 c2%_no] M1] | [[[St - M1] c2%_no] [St - M1]]]] & [[[M2 c2%_no] M2] | [[[St - M2] c2%_no] [St - M2]]]];
define c1%_look%_m [[[[[L2 & M2] c1%_look%_m] [L1 & [St - L2]]] & [[[M1 c1%_look%_m] M1] | [[[St - M1] c1%_look%_m] [St - M1]]]] & [[[M2 c1%_look%_m] M2] | [[[St - M2] c1%_look%_m] [St - M2]]]];
define c1%_look%_f [[[[[L2 & [St - M2]] c1%_look%_f] [L1 & [St - L2]]] & [[[M1 c1%_look%_f] M1] | [[[St - M1] c1%_look%_f] [St - M1]]]] & [[[M2 c1%_look%_f] M2] | [[[St - M2] c1%_look%_f] [St - M2]]]];
define c2%_look%_m [[[[[L1 & M1] c2%_look%_m] [L2 & [St - L1]]] & [[[M1 c2%_look%_m] M1] | [[[St - M1] c2%_look%_m] [St - M1]]]] & [[[M2 c2%_look%_m] M2] | [[[St - M2] c2%_look%_m] [St - M2]]]];
define c2%_look%_f [[[[[L1 & [St - M1]] c2%_look%_f] [L2 & [St - L1]]] & [[[M1 c2%_look%_f] M1] | [[[St - M1] c2%_look%_f] [St - M1]]]] & [[[M2 c2%_look%_f] M2] | [[[St - M2] c2%_look%_f] [St - M2]]]];
define Event [c1%_yes | c1%_no | c2%_yes | c2%_no | c1%_look%_m | c1%_look%_f | c2%_look%_m | c2%_look%_f];
source kat.fst
define c1 RelKst([[c1%_yes .o. [Event .x. Event] .o. c1%_yes] | [[c1%_no .o. [Event .x. Event] .o. c1%_no] | [[c1%_look%_m .o. [Event .x. Event] .o. c1%_look%_m] | [[c1%_look%_f .o. [Event .x. Event] .o. c1%_look%_f] | [[c2%_yes .o. [Event .x. Event] .o. c2%_yes] | [[c2%_no .o. [Event .x. Event] .o. c2%_no] | [[c2%_look%_m .o. [Event .x. Event] .o. [c2%_look%_m | c2%_look%_f]] | [c2%_look%_f .o. [Event .x. Event] .o. [c2%_look%_m | c2%_look%_f]]]]]]]]]);
define c2 RelKst([[c1%_yes .o. [Event .x. Event] .o. c1%_yes] | [[c1%_no .o. [Event .x. Event] .o. c1%_no] | [[c1%_look%_m .o. [Event .x. Event] .o. [c1%_look%_m | c1%_look%_f]] | [[c1%_look%_f .o. [Event .x. Event] .o. [c1%_look%_m | c1%_look%_f]] | [[c2%_yes .o. [Event .x. Event] .o. c2%_yes] | [[c2%_no .o. [Event .x. Event] .o. c2%_no] | [[c2%_look%_m .o. [Event .x. Event] .o. c2%_look%_m] | [c2%_look%_f .o. [Event .x. Event] .o. c2%_look%_f]]]]]]]]);

# 6
define W0 L2;
# 36
define amy0 W0 .o. c1;
# 36
define bob0 W0 .o. c2;

define W1 Cn(W0, [c1%_look%_m | c1%_look%_f]);
define W2 Cn(W1, [c2%_look%_m | c2%_look%_f]);

# Kripke relations restricted to W1 and W2
define amy1 W1 .o. c1 .o. W1;
define bob1 W1 .o. c2 .o. W1;
define amy2 W2 .o. c1 .o. W2;
define bob2 W2 .o. c2 .o. W2;

# In W1, Amy has picked up the muddiness of the other agent, and Bob not.
# This is illustrated by their alternatives in the world where both are muddy.
# Amy has eliminated [1 0], and Bob not.

# regex [Cn([W0 & M1 & M2],W1) .o. amy1].l;
# 1 1 0 1 c1_look_m 1 1 1 0
# 0 1 0 1 c1_look_m 0 1 1 0
# regex [Cn([W0 & M1 & M2],W1) .o. bob1].l;
# 1 1 0 1 c1_look_m 1 1 1 0
# 1 0 0 1 c1_look_f 1 0 1 0
# 0 1 0 1 c1_look_m 0 1 1 0

# W2 is symmetric, both have sensed the muddiness of the other agent.
# regex [Cn([W0 & M1 & M2],W2) .o. amy2].l;
# 1 1 0 1 c1_look_m 1 1 1 0 c2_look_m 1 1 0 1
# 0 1 0 1 c1_look_m 0 1 1 0 c2_look_f 0 1 0 1
# regex [Cn([W0 & M1 & M2],W2) .o. bob2].l;
# 1 1 0 1 c1_look_m 1 1 1 0 c2_look_m 1 1 0 1
# 1 0 0 1 c1_look_f 1 0 1 0 c2_look_m 1 0 0 1

# The next step requires WH and know-whether operators. These refer to a set of worlds
# as well as a Kripke relation.

# WH operator forming a different-answer relation, the complement of the
# partition-semantics denotation.
define WH(Y,X) [[X & Y] .x. [X - Y]] | [[X - Y] .x. [X & Y]];


# Know Q. Subtract away where we can get using R to cells that are
# different according to Q. Q should be the complement of the
# partition-semantics equivalence relation.
define Kw(Q,R,X) X - [[R.i .o. Q .o. R] & X];

# Agent R knows fluent F in space of worlds X.
define Know(R,F,X) Kw(WH(Cn(X,F),X),R,X);

# Test it in W2.
# Amy knows she is muddy iff Bob isn't muddy.
# regex Know(amy2,M1,W2);
# 1 0 0 1 c1_look_f 1 0 1 0 c2_look_m 1 0 0 1

# Know(amy2,M2,W2);
# She knows whether bob is muddy in all the worlds.
# 1 1 0 1 c1_look_m 1 1 1 0 c2_look_m 1 1 0 1
# 1 0 0 1 c1_look_f 1 0 1 0 c2_look_m 1 0 0 1
# 0 1 0 1 c1_look_m 0 1 1 0 c2_look_f 0 1 0 1

# We get symmetric results for Bob.

# Now Amy talks. She says yes if she knows whether she is muddy, and otherwise no.
# This is an epistemic precondition for how he answers, and her answer is deterministic.
define W3 Cn(Know(amy2,M1,W2),c1%_yes) | Cn([W2 - Know(amy2,M1,W2)],c1%_no);

# Define amy3 and bob3.  It looks like we don't need this, it proceeds by restriction.
define amy3 W3 .o. c1 .o. W3;
define bob3 W3 .o. c2 .o. W3;

# Bob talks.
define W4 Cn(Know(bob3,M2,W3),c2%_yes) | Cn([W3 - Know(bob3,M2,W3)],c2%_no);
# 1 1 0 1 c1_look_m 1 1 1 0 c2_look_m 1 1 0 1 c1_no 1 1 1 0 c2_yes 1 1 0 1
# 1 0 0 1 c1_look_f 1 0 1 0 c2_look_m 1 0 0 1 c1_yes 1 0 1 0 c2_yes 1 0 0 1
# 0 1 0 1 c1_look_m 0 1 1 0 c2_look_f 0 1 0 1 c1_no 0 1 1 0 c2_yes 0 1 0 1
# He says yes in every world. So poor Amy gets no information!

# Define the Kripke relations for this step.
define amy4 W4 .o. c1 .o. W4;
define bob4 W4 .o. c2 .o. W4;

# Amy talks again.
define W5 Cn(Know(amy4,M1,W4),c1%_yes) | Cn([W4 - Know(amy4,M1,W4)],c1%_no);

# She answers as before, reflecting that she learned nothing from Bob's statement.
# 0 1 0 1 c1_look_m 0 1 1 0 c2_look_f 0 1 0 1 c1_no 0 1 1 0 c2_yes 0 1 0 1 c1_no 0 1 1 0
# 1 1 0 1 c1_look_m 1 1 1 0 c2_look_m 1 1 0 1 c1_no 1 1 1 0 c2_yes 1 1 0 1 c1_no 1 1 1 0
# 1 0 0 1 c1_look_f 1 0 1 0 c2_look_m 1 0 0 1 c1_yes 1 0 1 0 c2_yes 1 0 0 1 c1_yes 1 0 1 0

# Define the Kripke relations for this step.
define amy5 W5 .o. c1 .o. W5;
define bob5 W5 .o. c2 .o. W5;

# Bob talks again. Nothing is gonna change.
define W6 Cn(Know(bob5,M2,W5),c2%_yes) | Cn([W5 - Know(bob5,M2,W5)],c2%_no);
# 1 1 0 1 c1_look_m 1 1 1 0 c2_look_m 1 1 0 1 c1_no 1 1 1 0 c2_yes 1 1 0 1 c1_no 1 1 1 0 c2_yes 1 1 0 1
# 1 0 0 1 c1_look_f 1 0 1 0 c2_look_m 1 0 0 1 c1_yes 1 0 1 0 c2_yes 1 0 0 1 c1_yes 1 0 1 0 c2_yes 1 0 0 1
# 0 1 0 1 c1_look_m 0 1 1 0 c2_look_f 0 1 0 1 c1_no 0 1 1 0 c2_yes 0 1 0 1 c1_no 0 1 1 0 c2_yes 0 1 0 1
