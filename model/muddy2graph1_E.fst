# Load this after muddy2graph1.fsu.

# This does various sanity checks and investigations
# on events and states.

# There are this many well formed states.
# regex St;
# 1.5 Kb. 16 states, 29 arcs, 12288 paths
# (All all of them reachable?)

# Define the event pairs.
define al al1 | al0;
define bl bl1 | bl0;

define ar ar1 | ar0;
define br br1 | br0;

define as as1 | as0;
define bs bs1 | bs0;
# That's it.

# In any state, exactly one event from a pair can happen. Check this by checking
# the counts. The count is 12288 in each case.

regex Cn(St,al);
regex Cn(St,bl);
regex Cn(St,ar);
regex Cn(St,br);
regex Cn(St,as);
regex Cn(St,bs);

# Relation that deletes all but the end state.
# The "?" covers a bare event.
define DeleteStart0 [[St ?]* -> 0 || .#. _ St .#.];

# The same as an operator that delivers the end states.
define EndState(X) [X .o. DeleteStart0].l;

# EndState(Event) 9987 paths. It does not include all states.

# Initial states. This has just three states.
define I0 Ea12 & Ea23 & Ea31 & Eb12 & Eb23 & Eb31 & Nst(Ra1) & Nst(Ra2) & Nst(Ra3) & Nst(Rb1) & Nst(Rb2) & Nst(Rb3);

# Worlds that start with an initial state of the above kind.
# This is a more restricted modal space.
define V Cn(I0,W);

# EndState(V)  This has only 114 states. It correponds to 38 graphs for a given base world.
define Ie EndState(V);

# Test that I0 is included.
$ regex I0 - Ie; => 0 paths.

# Accessibility on states
# MB + MA + Ea12 + Ea23 + Ea31 + Eb12 + Eb23 + Eb31 + Ra1 + Ra2 + Ra3 + Rb1 + Rb2 + Rb3
# To start assume the edge fluents stay constant.
# What about reflection fluents?

# Relation on Ie that is constant on Ea12, and similarly for the other edges
# and other agent.
define cEa12 Ie .o. [[Ea12 .x. Ea12] | [Nst(Ea12) .x. Nst(Ea12)]] .o. Ie;
define cEa23 Ie .o. [[Ea23 .x. Ea23] | [Nst(Ea23) .x. Nst(Ea23)]] .o. Ie;
define cEa31 Ie .o. [[Ea31 .x. Ea31] | [Nst(Ea31) .x. Nst(Ea31)]] .o. Ie;

define cEb12 Ie .o. [[Eb12 .x. Eb12] | [Nst(Eb12) .x. Nst(Eb12)]] .o. Ie;
define cEb23 Ie .o. [[Eb23 .x. Eb23] | [Nst(Eb23) .x. Nst(Eb23)]] .o. Ie;
define cEb31 Ie .o. [[Eb31 .x. Eb31] | [Nst(Eb31) .x. Nst(Eb31)]] .o. Ie;

define ConstantEdge cEa12 & cEa23 & cEa31 & cEb12 & cEb23 & cEb31;

# [Nst(MB) & MA & Ea12] .o. ConstantEdge
# Similarly for reflection, in case it it needed.
# Are reflection bits constant across alternatives though?

define cRa1 Ie .o. [[Ra1 .x. Ra1] | [Nst(Ra1) .x. Nst(Ra1)]] .o. Ie;
define cRa2 Ie .o. [[Ra2 .x. Ra2] | [Nst(Ra2) .x. Nst(Ra2)]] .o. Ie;
define cRa3 Ie .o. [[Ra3 .x. Ra3] | [Nst(Ra3) .x. Nst(Ra3)]] .o. Ie;
define cRb1 Ie .o. [[Rb1 .x. Rb1] | [Nst(Rb1) .x. Nst(Rb1)]] .o. Ie;
define cRb2 Ie .o. [[Rb2 .x. Rb2] | [Nst(Rb2) .x. Nst(Rb2)]] .o. Ie;
define cRb3 Ie .o. [[Rb3 .x. Rb3] | [Nst(Rb3) .x. Nst(Rb3)]] .o. Ie;

define ConstantReflection cRa1 & cRa2 & cRa3 & cRb1 & cRb2 & cRb3;
define ConstantBoth ConstantEdge & ConstantReflection;

# [Nst(MB) & MA & Ea12] .o. ConstantEdge .o. [MB & Nst(MA)];
# Should be expect amy to be definable from the graph?
