
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

-- Ea12 and Ea31 vary the values of MA
-- Ea23 has a constant value for MA
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

-- Similarly for Bob.
-- Eb12 and Eb31 vary the values of MB.
-- Eb23 has a constant value for MB
action br1 = (constant MB) & (constant MA) & 
   (constant Eb12) & (constant Eb23) & (constant Eb31) & (constant Ea12) & (constant Ea23) & (constant Ea31) &
   (constant Ra1) & (constant Ra2) & (constant Ra3) &
   (((prezero Eb12) & (prezero Eb31) & (postone Rb1)) + (((preone Eb12) + (preone Eb31)) & (postzero Rb1))) &
   (((prezero Eb12) & (postone Rb2)) + ((preone Eb12) & (postzero Rb2))) &
   (((prezero Eb31) & (postone Rb3)) + ((preone Eb31) & (postzero Rb3))) &
   (((postone MA) & (postzero MB) & (postone Rb1)) +
   ((postzero MA) & (postone MB) & (postone Rb2)) +
   ((postone MA) & (postone MB) & (postone Rb3)))


action br0 = (constant MB) & (constant MA) & 
   (constant Eb12) & (constant Eb23) & (constant Eb31) & (constant Ea12) & (constant Ea23) & (constant Ea31) &
   (constant Ra1) & (constant Ra2) & (constant Ra3) &
   (((prezero Eb12) & (prezero Eb31) & (postone Rb1)) + (((preone Eb12) + (preone Eb31)) & (postzero Rb1))) &
   (((prezero Eb12) & (postone Rb2)) + ((preone Eb12) & (postzero Rb2))) &
   (((prezero Eb31) & (postone Rb3)) + ((preone Eb31) & (postzero Rb3))) &
   (((postone MA) & (postzero MB) & (postzero Rb1)) +
   ((postzero MA) & (postone MB) & (postzero Rb2)) +
   ((postone MA) & (postone MB) & (postzero Rb3)))

-- Saying actions
-- Assume Amy edges are constant when Amy speaks.
-- Bob edges should be cut if Amy says different things at the vertices. This requires
-- looking at the Ra fluents.
action as1 = (constant MB) & (constant MA) & 
   (constant Ra1) & (constant Ra2) & (constant Ra3) & (constant Rb1) & (constant Rb2) & (constant Rb3) &
   (constant Ea12) & (constant Ea23) & (constant Ea31) &
  (((postone Eb12) & (preone Eb12) & (presame Ra1 Ra2)) + ((postzero Eb12) & (prezero Eb12)) + ((postzero Eb12) & (predifferent Ra1 Ra2))) &
  (((postone Eb23) & (preone Eb23) & (presame Ra2 Ra3)) + ((postzero Eb23) & (prezero Eb23)) + ((postzero Eb23) & (predifferent Ra2 Ra3))) &
  (((postone Eb31) & (preone Eb31) & (presame Ra3 Ra1)) + ((postzero Eb32) & (prezero Eb31)) + ((postzero Eb31) & (predifferent Ra3 Ra1))) &
   (((preone MA) & (prezero MB) & (preone Ra1)) +
    ((prezero MA) & (preone MB) & (preone Ra2)) +
    ((preone MA) & (preone MB) & (preone Ra3)))
-- The last part identifes the base worlds in which as1 can happen.  

-- Amy say no.
action as0 = (constant MB) & (constant MA) & 
   (constant Ra1) & (constant Ra2) & (constant Ra3) & (constant Rb1) & (constant Rb2) & (constant Rb3) &
   (constant Ea12) & (constant Ea23) & (constant Ea31) &
  (((postone Eb12) & (preone Eb12) & (presame Ra1 Ra2)) + ((postzero Eb12) & (prezero Eb12)) + ((postzero Eb12) & (predifferent Ra1 Ra2))) &
  (((postone Eb23) & (preone Eb23) & (presame Ra2 Ra3)) + ((postzero Eb23) & (prezero Eb23)) + ((postzero Eb23) & (predifferent Ra2 Ra3))) &
  (((postone Eb31) & (preone Eb31) & (presame Ra3 Ra1)) + ((postzero Eb32) & (prezero Eb31)) + ((postzero Eb31) & (predifferent Ra3 Ra1))) &
   (((preone MA) & (prezero MB) & (prezero Ra1)) +
    ((prezero MA) & (preone MB) & (prezero Ra2)) +
    ((preone MA) & (preone MB) & (prezero Ra3)))
-- This differs from as1 only in the last three lines, which identify the base worlds wher as0 can happen.

-- Bob saying actions.
-- Assume Bob edges are constant.
-- Amy edges should be cut if Bob says different things at the vertices. This requires
-- looking at the Rb fluents.

action bs1 = (constant MB) & (constant MA) & 
   (constant Ra1) & (constant Ra2) & (constant Ra3) & (constant Rb1) & (constant Rb2) & (constant Rb3) &
   (constant Eb12) & (constant Eb23) & (constant Eb31) &
  (((postone Ea12) & (preone Ea12) & (presame Rb1 Rb2)) + ((postzero Ea12) & (prezero Ea12)) + ((postzero Ea12) & (predifferent Rb1 Rb2))) &
  (((postone Ea23) & (preone Ea23) & (presame Rb2 Rb3)) + ((postzero Ea23) & (prezero Ea23)) + ((postzero Ea23) & (predifferent Rb2 Rb3))) &
  (((postone Ea31) & (preone Ea31) & (presame Rb3 Rb1)) + ((postzero Ea31) & (prezero Ea31)) + ((postzero Ea31) & (predifferent Rb3 Rb1))) &
   (((preone MA) & (prezero MB) & (preone Rb1)) +
    ((prezero MA) & (preone MB) & (preone Rb2)) +
    ((preone MA) & (preone MB) & (preone Rb3)))

-- This glosses the principle for updating Ea1.
--   (((postone Ea12) & (preone Ea12) & (presame Rb1 Rb2)) Ea12 is in the out state if it was in the in state, 
--                                                         and Bob is going to say the same thing in 1 and 2.
--   ((postzero Ea12) & (prezero Ea12))                    Ea12 is not in the out state if it was not in the in state.
--   ((postzero Ea12) & (predifferent Rb1 Rb2))) &         Ea12 is not in the out state if Bob is going to say different things
--                                                         in 1 and 2.

action bs0 = (constant MB) & (constant MA) & 
   (constant Ra1) & (constant Ra2) & (constant Ra3) & (constant Rb1) & (constant Rb2) & (constant Rb3) &
   (constant Eb12) & (constant Eb23) & (constant Eb31) &
  (((postone Ea12) & (preone Ea12) & (presame Rb1 Rb2)) + ((postzero Ea12) & (prezero Ea12)) + ((postzero Ea12) & (predifferent Rb1 Rb2))) &
  (((postone Ea23) & (preone Ea23) & (presame Rb2 Rb3)) + ((postzero Ea23) & (prezero Ea23)) + ((postzero Ea23) & (predifferent Rb2 Rb3))) &
  (((postone Ea31) & (preone Ea31) & (presame Rb3 Rb1)) + ((postzero Ea31) & (prezero Ea31)) + ((postzero Ea31) & (predifferent Rb3 Rb1))) &
   (((preone MA) & (prezero MB) & (prezero Rb1)) +
    ((prezero MA) & (preone MB) & (prezero Rb2)) +
    ((preone MA) & (preone MB) & (prezero Rb3)))

agent amy =
  (al1 -> al1)
  (al0 -> al0)
  (bl1 -> bl1 + bl0)
  (bl0 -> bl1 + bl0)
  (ar1 -> ar1)
  (ar0 -> ar0)
  (br1 -> br1 + br0)
  (br0 -> br1 + br0)
  (as1 -> as1)
  (as0 -> as0)
  (bs1 -> bs1)
  (bs0 -> bs0)
  
agent bob =
  (al1 -> al1 + al0)
  (al0 -> al1 + al0)
  (bl1 -> bl1)
  (bl0 -> bl0)
  (ar1 -> ar1 + ar0)
  (ar0 -> ar1 + ar0)
  (br1 -> br1)
  (br0 -> br0)
  (as1 -> as1)
  (as0 -> as0)
  (bs1 -> bs1)
  (bs0 -> bs0)
  
queryall al1




