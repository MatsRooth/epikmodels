###########################################
#
#  KAT in fst
#  Mats Rooth
#
###########################################


## This should be loaded in a context that has these variables defined.
#  St --Boolean vectors such as 0 1 1 0. The length is the number of fluents.
#  UnequalStPair --Non-matching vectors such as 0 1 1 0 1 1 1. They differ
#   in one or more positions.
#  Event --Decorated events such as H1 peek%_amy%_H1.

# Reduce a sequence of two tests to a single test, by deleting the second one.
# define Squash St -> 0 || St _;

# String that doesn't contain a non-matching test pair.
# define Wf0 ~[$ UnequalStPair];

# With 14 fluents, this is not huge, but it takes 20 min to compile
# 19.7 Mb. 192525 states, 565285 arcs, Circular.
# Save time like this. It defines Wf0 and Squash.
# The replacement gave odd results, try it again.
echo Start load defined unequal14.net
load defined model/unequal14.net

echo finished load defined

# KAT concatenation operation on sets of histories.
# Concatenate in the string algebra, eliminate strings with
# non-matching tests, then map to guarded strings.
define Cn(X,Y) [[[X Y] & Wf0] .o. Squash].l;
#                ---- concatenate in the string algebra
#                     ----- eliminate strings that contain non-matching tests
#                             ---------- map to a guarded string

# Remove medial Booleans.
# This is can be used to produce a shorter print name.
# It isn't used in the analysis.
define Short0 p -> 0 || Event0 _ Event0;
define Short(X) [X .o. Short0].l;

# Kleene Plus
define Kpl(X) [[[X+] & Wf0] .o. Squash].l;
#              --- Kleene plus in the string algebra
#                  ----- eliminate strings that contain non-matching tests
#                             ---------- map to a guarded string

# Kleene Star
# The identity is St, not the empty string.
define Kst(X) St |  Kpl(X);

# KAT concatenation operation on relations
define Cnr(R,S) Squash.i .o. Wf0 .o. [R S] .o. Wf0 .o. Squash;
#                                    ---- concatenate in the string algebra
#                            ------        ------- constrain domain and codomain
#                                                  to contain non non-matching tests
#               ------------                       ---------- map to guarded strings
#                                                             on both sides.

# Kleene plus on relations
define RelKpl(X) Squash.i .o. Wf0 .o. [X+] .o. Wf0 .o. Squash;
#                                  ----   Kleene plus in the string algebra
#                          ------       -------    constrain domain and codomain
#                                                  to contain non non-matching tests
#             ------------                         ---------- map to guarded strings
#                                                             on both sides.

# Kleene star on relations
# This is used in defining world alternative relations.
# The total relation on St is included.
define RelKst(X) [St .x. St] | RelKpl(X); 

###########################################
#
#  Worlds
#
##########################################

# Worlds (or histories) are the Kleene closure of the events.
# Since the operation Kpl is used, pre-conditions are enforced.
# Include states without a following event.
define W St | Kpl(Event);

########################################################
#
#  Complement in W
#
########################################################

define Not(X) W - X;

########################################################
#
#  Insert tests in a string of events to make a world
#
########################################################

# This can be used to insert tests appropriately in an event
# sequence, e.g.
#   World({cd}) = H c H d H.
# It's not always a function. 

define World(X) [X .o. [0 -> St] .o. W].l;

########################################################
#
#  Dress a bare event
#
########################################################

# define Event(Y) [St Y St] & W;

########################################################
#
#  Diamond modality
#
########################################################

# It's algebraic modal logic. The diamond modality is pre-image.

# R is a Kripke relation on W
# X is a proposition

define Dia(R,X) [R .o. X].u;

# In the application, R is named by an agent name, e.g.
# Dia(bob,X) where X is a propositional term.

########################################################
#
#  Box modality
#
########################################################

# It's the dual of diamond.

define Box(R,X) Not(Dia(R,Not(X)));

########################################################
#
#  Short worlds
#
########################################################

# define One Event;
# define Two Cn(One,Event);
# define Three Cn(Two,Event);
# define Four Cn(Three,Event);
# define Five Cn(Four,Event);
