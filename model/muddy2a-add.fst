source muddy2a.fst

# There are three states defined by Amy's muddyness (least significant position) and Bob's.
# 0 1
# 1 0
# 1 1

# Reading the above as binary numbers, disignate these states as S1, S2, S2.
define S1 MA & Nst(MB);  # 01
define S2 Nst(MA) & MB;  # 10
define S3 MA & MB;       # 11


# There are then 2^3 state propositions, using bit vectors.
# Minimal information.
# P111 [1 1 1]; # S1, S2, and S3 are a possibility.

# Medial information
# P110 [1 1 "0"];
# P101 [1 "0" 1];
# P100 [1 "0" "0"];

# Maximal information
# P001 ["0" "0" 1]; # Only S1 is a possibility.
# P010 ["0" 1 "0"]; # Only S2 is a possibility.
# P100 [1 "0" "0"]; # Only S3 is a possibility.

# The above are coded with the new fluents P1, P2, P3, which are interpreted as follows.
# P1 -- S1 is a possibility
# P2 -- S2 is a possibility
# P3 -- S3 is a possibility

# So P001 is P1 & Nst(P2) & Nst(P3).


# The initial state is going to be [St P111]. The muddiness can be anything, and there is no information.

# An event moves the epistemic vector along the information order towards more information, by eleminating
# possibilities (turning a 1 to 0).
# For instance amy_loo_1 sets P2 to 0, leaving values of P1 and P3 the same.

# What does bob_loo_1 do for Amy's P in the base world? For Amy it can be
# bob_loo_1 or bob_loo_0, so it should not change the P vector on the base side.
# This is indicating that the P-vectors are not going do not the Kripke relations,
# even though they encode some of the same information.

########################
# This is the old definition of amy_loo_1, focusing on amy.
# action amy_loo_1 = (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB)))
# In the agent definition for Amy
#    (amy_loo_1 -> amy_loo_1)

# Now we have new fluents P1, P2, P3. P2 is supposed to be set to 0, and P1 and P3 preserved. Add these conjuncts.

# (((test P1);id;(test P1))+((~(test P1));id;(~(test P1))))  preserve P1
# (((test P3);id;(test P3))+((~(test P3));id;(~(test P3))))  preserve P3
# (((test P2) + (~test P2));id;(~(test P2)))  set P2 to false

# This is the old definition of amy_say_1, which simply preserves the extensional state.
# action amy_say_1 = (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))

# To this is added a quality pre-condition.  Amy should say 1 only if she considers world S2 (where she is not muddy) impossible.
# So the precondition is ~test P2.  The other P's are preserved. These conjuncts are added.

# (((test P1);id;(test P1))+((~(test P1));id;(~(test P1))))  preserve P1
# (((test P3);id;(test P3))+((~(test P3));id;(~(test P3))))  preserve P3
# (~test P2);id;(~test P2)  Precondition that P2, amounting to S2 (or a world of state type S2) not being an alternative.