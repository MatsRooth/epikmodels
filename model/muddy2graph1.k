-- Encode a graph structure for two muddy children in fluents, by indicating the presence or absence of edges.
-- Vertices (corresponding to combinations of MB and MA in earlier analysis)
--  01 1 Amy is muddy, Bob is not muddy.
--  10 2 Amy is not muddy, Bob is muddy.
--  11 3 Amy and Bob are muddy.

-- Edges
-- Ea12  Amy edge between 1 and 2. Vertices 1 and 2 are in the same cell in Amy's partition.
-- Ea23  similar
-- Ea31  similar
-- Eb12  Bob edge between 1 and 2. Vertices 1 and 2 are in the same cell in Bob's partition.
-- Eb23  similar
-- Eb31  similar

-- Reflection bits
-- Ra1  In world 01, Amy has reflected that she knows whether she is muddy. Amy will say yes if she speaks in 01.
-- Ra2  similar
-- Ra3  similar
-- Rb1  In world 01, Bob has reflected that he knows whether he is muddy. Bob will say yes if she speaks in 01.
-- Rb2  similar
-- Rb3  similar

-- To this we add MB and MA. 
-- A state generator is a vector of 14 bits
world = MB + MA + Ea12 + Ea23 + Ea31 + Eb12 + Eb23 + Eb31 + Ra1 + Ra2 + Ra3 + Rb1 + Rb2 + Rb3

-- At least one child is muddy.
assert MB + MA

-- Operators for describing pre- and post-conditions of actions.
-- constant F Value of F remains constant.
-- notup F Value of F does not go from 0 to 1. This always applies to edges. Equivalent to constant(F) + (test F);id;(~test F).
-- postzero F F is false in the postcondition. This is used for eliminating edges.
-- postone F
-- postzero F
-- prezero F

-- Foe all actions
-- (constant MB) & (constant MA) & 
-- (notup Ea12) & (notup Ea23) & (constant Ea31) & (constant Eb12) & (constant Eb23) & (constant Eb31) &
-- (notdown Ra1) & (notdown Ra2) & (notdown Ra3) & (notdown Rb1) & (notdown Rb2) & (notdown Rb3)

-- Amy looks at 1 (Bob muddy).
-- This amounts to Amy sensing MB.
-- This eliminates edges joining vertices where the value of MB differs.
-- This characterizes Amy looking in general.
action al1 = (constant MB) & (constant MA) & 
   (constant Ra1) & (constant Ra2) & (constant Ra3) & (constant Rb1) & (constant Rb2) & (constant Rb3) &
   (postzero Ea12) & (constant Ea23) & (postzero Ea31) & (constant Eb12) & (constant Eb23) & (constant Eb31) &
   (preone MB)

-- Amy looks at 0 (Bob not muddy).
-- This is the same in its action on the graph.  It differs in the precondition about MB.
action al0 = (constant MB) & (constant MA) & 
   (constant Ra1) & (constant Ra2) & (constant Ra3) & (constant Rb1) & (constant Rb2) & (constant Rb3) &
   (postzero Ea12) & (constant Ea23) & (postzero Ea31) & (constant Eb12) & (constant Eb23) & (constant Eb31) &
   (prezero MB)

-- Bob looks at 1 (Amy muddy).
-- This amounts to Bob sensing MA.
-- This eliminates edges joining vertices where the value of MA differs, namely 
action bl1 = (constant MB) & (constant MA) & 
   (constant Ra1) & (constant Ra2) & (constant Ra3) & (constant Rb1) & (constant Rb2) & (constant Rb3) &
   (constant Ea12) & (constant Ea23) & (constant Ea31) & (postzero Eb12) & (postzero Eb23) & (constant Eb31) &
   (preone MA)

-- Bob looks at 0 (Amy not muddy).
action bl0 = (constant MB) & (constant MA) & 
   (constant Ra1) & (constant Ra2) & (constant Ra3) & (constant Rb1) & (constant Rb2) & (constant Rb3) &
   (constant Ea12) & (constant Ea23) & (constant Ea31) & (postzero Eb12) & (postzero Eb23) & (constant Eb31) &
   (prezero MA)

-- Amy reflects 1. This sets the Ra fluents to 1 for worlds where the alternatives (as encoded in the graph) have
-- a consistent answer to the question whether MA.
-- E fluents (encoding the graph) are constant.
action ar1 = (constant MB) & (constant MA) & 
   (constant Eb12) & (constant Eb23) & (constant Eb31) & (constant Ea12) & (constant Ea23) & (constant Ea31) &
   (constant Rb1) & (constant Rb2) & (constant Rb3) &
   (((prezero Ea12) & (postone Ra1)) + ((preone Ea12) & (postzero Ra1))) &
   (((prezero Ea12) & (prezero Ea23) & (postone Ra2)) + (((preone Ea12) + (preone Ea23)) & (postzero Ra2))) &
   (((prezero Ea23) & (postone Ra3)) + ((preone Ea23) & (postzero Ra3))) &
   (((postone MA) & (postzero MB) & (postone Ra1)) +
   ((postzero MA) & (postone MB) & (postone Ra2)) +
   ((postone MA) & (postone MB) & (postone Ra3)))


-- Ea1 and Ea2 vary the values of MA
-- Ea3 has a constant value for MA
-- Ra1 should go to 1 if no varying Amy edge is incident on vertex 1, i.e. if Ea12 is not present.
-- If Ea12 is present, it should go to 0. It amounts to this event description.
-- (((prezero Ea12) & (postone Ra1)) + ((preone Ea12) & (postzero Ra1)))
--
-- Ra2 should go to 1 if no varying Amy edge is incident on vertex 2, i.e. if neither Ea12 nor Ea23 is present.
-- If Ea12 or Ea23 is present, it should go to 0.
-- (((prezero Ea12) & (prezero Ea23) & (postone Ra2)) + (((preone Ea12) + (preone Ea23)) & (postzero Ra2)))
--
-- Ra3 should go to 1 if no varying Amy edge is incident on vertex 1, i.e. if Ea23 is not present.
-- If Ea23 is present, it should go to 0.
-- (((prezero Ea12) & (postone Ra3)) + ((preone Ea12) & (postzero Ra3)))

action ar0 = (constant MB) & (constant MA) & 
   (constant Eb12) & (constant Eb23) & (constant Eb31) & (constant Ea12) & (constant Ea23) & (constant Ea31) &
   (constant Rb1) & (constant Rb2) & (constant Rb3) &
   (((prezero Ea12) & (postone Ra1)) + ((preone Ea12) & (postzero Ra1))) &
   (((prezero Ea12) & (prezero Ea23) & (postone Ra2)) + (((preone Ea12) + (preone Ea23)) & (postzero Ra2))) &
   (((prezero Ea23) & (postone Ra3)) + ((preone Ea23) & (postzero Ra3))) &
   (((postone MA) & (postzero MB) & (postzero Ra1)) +
   ((postzero MA) & (postone MB) & (postzero Ra2)) +
   ((postone MA) & (postone MB) & (postzero Ra3)))

agent amy =
  (al1 -> al1)
  (al0 -> al0)
  (bl1 -> bl1 + bl0)
  (bl0 -> bl1 + bl0)
  (ar1 -> ar1)
  (ar0 -> ar0)
  (br1 -> br1 + br0)
  (br0 -> br1 + br0)

agent bob =
  (al1 -> al1 + al0)
  (al0 -> al1 + al0)
  (bl1 -> bl1)
  (bl0 -> bl0)
  (ar1 -> ar1 + ar0)
  (ar0 -> ar1 + ar0)
  (br1 -> br1)
  (br0 -> br0)

queryall  MA & Nst(MB) & Ea12 & Ea23 & Ea31 & Eb12 & Eb23 & Eb31 & Nst(Ra1) & Nst(Ra2) & Nst(Ra3) & Nst(Rb1) & Nst(Rb2) & Nst(Rb3);


