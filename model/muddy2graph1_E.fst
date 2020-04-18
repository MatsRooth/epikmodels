# Load this after muddy2graph1_E.fsu.

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
define W0 Cn(I0,W);

# EndState(W0)  This has only 114 states. It correponds to 38 graphs for a given base world.




