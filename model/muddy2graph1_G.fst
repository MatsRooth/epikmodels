# Initial graph state G0.
define G0 MA & Nst(MB) & Ea12 & Ea23 & Ea31 & Eb12 & Eb23 & Eb31 & Nst(Ra1) & Nst(Ra2) & Nst(Ra3) & Nst(Rb1) & Nst(Rb2) & Nst(Rb3);
regex G0;
print words
# 0 1 1 1 1 1 1 1 0 0 0 0 0 0

define G1 Cn(G0,al0);
regex G1;
print words
# 0 1 1 1 1 1 1 1 0 0 0 0 0 0 al0 0 1 0 1 0 1 1 1 0 0 0 0 0 0

define G2 Cn(G1,bl1);
regex G2;
print words
# 0 1 1 1 1 1 1 1 0 0 0 0 0 0 al0 0 1 0 1 0 1 1 1 0 0 0 0 0 0 bl1 0 1 0 1 0 0 0 1 0 0 0 0 0 0

# These are correct. Next come ar (Aly reflect) and br (Bob reflect). These need to be defined
# in the k file.

define G3 Cn(G2,ar1);
regex G3;
print words
# 0 1 1 1 1 1 1 1 0 0 0 0 0 0 al0 0 1 0 1 0 1 1 1 0 0 0 0 0 0 bl1 0 1 0 1 0 0 0 1 0 0 0 0 0 0 ar1 0 1 0 1 0 0 0 1 1 0 0 0 0 0

# This is correct, or at least as targeted.

# regex  Cn(G2,ar0);
# 80 bytes. 1 state, 0 arcs, 0 paths.
# Why? In world 01 at this stage, ar0 can not happen, because in that world, Amy has uniform alternatives on the question
# whether she is muddy.

define G4 Cn(G3,br0);
regex G4;
print words
#     0 1 1 1 1 1 1 1 0 0 0 0 0 0
# al0 0 1 0 1 0 1 1 1 0 0 0 0 0 0
# bl1 0 1 0 1 0 0 0 1 0 0 0 0 0 0
# ar1 0 1 0 1 0 0 0 1 1 0 0 0 0 0
# br0 0 1 0 1 0 0 0 1 1 0 0 0 1 0

define G5 Cn(G4,as1);
regex G5;
print words

# 0 1 1 1 1 1 1 1 0 0 0 0 0 0
# al0 0 1  0 1 0 1 1 1  0 0 0 0 0 0
# bl1 0 1  0 1 0 0 0 1  0 0 0 0 0 0
# ar1 0 1  0 1 0 0 0 1  1 0 0 0 0 0
# br0 0 1  0 1 0 0 0 1  1 0 0 0 1 0
# as1 0 1  0 1 0 0 0 0  1 0 0 0 1 0

define G6 Cn(G5,bs0);
regex G6;
print words

define G7 Cn(G6,ar1);
regex G7;
print words

#     0 1 1 1 1 1 1 1 0 0 0 0 0 0
# al0 0 1 0 1 0 1 1 1 0 0 0 0 0 0
# bl1 0 1 0 1 0 0 0 1 0 0 0 0 0 0
# ar1 0 1 0 1 0 0 0 1 1 0 0 0 0 0
# br0 0 1 0 1 0 0 0 1 1 0 0 0 1 0
# as1 0 1 0 1 0 0 0 0 1 0 0 0 1 0
# bs0 0 1 0 0 0 0 0 0 1 0 0 0 1 0
# ar1 0 1 0 0 0 0 0 0 1 1 1 0 1 0


define G8 Cn(G7,br1);
regex G8;
print words

#     0 1 1 1 1 1 1 1 0 0 0 0 0 0
# al0 0 1 0 1 0 1 1 1 0 0 0 0 0 0
# bl1 0 1 0 1 0 0 0 1 0 0 0 0 0 0
# ar1 0 1 0 1 0 0 0 1 1 0 0 0 0 0
# br0 0 1 0 1 0 0 0 1 1 0 0 0 1 0
# as1 0 1 0 1 0 0 0 0 1 0 0 0 1 0
# bs0 0 1 0 0 0 0 0 0 1 0 0 0 1 0
# ar1 0 1 0 0 0 0 0 0 1 1 1 0 1 0
# br1 0 1 0 0 0 0 0 0 1 1 1 1 1 1

# The above agrees with the hand sheet.


