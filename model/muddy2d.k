-- Bob is muddy
-- Amy is muddy
-- Bob would say yes if he spoke
-- Amy would say yes if she spoke
world = MB+MA+WB+WA

-- At least one child is muddy.
assert (MA+MB)

-- Amy reflects that she knows whether she is muddy.
-- WA is 1 in the result.
-- Other fluents remain constant.
-- This does not impose the epistemic pre-condition.
action ar1 = (((test WA)+(~test WA));id;(test WA)) &
             (constant WB) &
	     (constant MA) & (constant MB)

-- Amy reflects that she doesn't know whether she is muddy.
action ar0 = (((test WA)+(~test WA));id;(~test WA)) &
             (constant WA) &
	     (constant MA) & (constant MB)

-- Bob reflects that he knows whether he is muddy.
action br1 = (((test WB)+(~test WB));id;(test WB)) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))

-- Bob reflects that he doesn't know whether he is muddy.
action br0 = (((test WB)+(~test WB));id;(~test WB)) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))

-- Amy says yes.
-- Check whether this has epistemic consequences for Bob.
action as1 = ((test WA);id;(test WA)) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))

-- Amy says no.
action as0 = ((~test WA);id;(~test WA)) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))

-- Bob says yes.
action bs1 = ((test WB);id;(test WB)) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))

-- Bob says no.
action bs0 = ((~test WB);id;(~test WB)) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))

-- Amy looks at 1 (Bob muddy).
-- This amounts to Amy sensing MB.
action al1 = ((test MB);id;(test MB)) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB)))) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB))))

-- Amy looks at 0 (Bob not muddy).
action al0 = ((~test MB);id;(~test MB)) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB)))) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB))))

-- Bob looks at 1 (Amy muddy).
-- Bob sensing MA.
action bl1 = ((test MA);id;(test MA)) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB)))) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB))))

-- Bob looks at 0 (Amy not muddy).
-- Bob sensing Nst(MA).
action bl0 = ((~test MA);id;(~test MA)) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB)))) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB))))


agent amy =
  (as1 -> as1)
  (as0 -> as0)
  (ar1 -> ar1)
  (ar0 -> ar0)
  (al1 -> al1)
  (al0 -> al0)
  (bs1 -> bs1)
  (bs0 -> bs0)
  (br1 -> br0 + br1)
  (br0 -> br0 + br1)
  (bl1 -> bl1 + bl0)
  (bl0 -> bl1 + bl0)


agent bob =
  (as1 -> as1)
  (as0 -> as0)
  (ar1 -> ar0 + ar1)
  (ar0 -> ar0 + ar1)
  (al1 -> al1 + al0)
  (al0 -> al1 + al0)
  (bs1 -> bs1)
  (bs0 -> bs0)
  (br1 -> br1)
  (br0 -> br0)
  (bl1 -> bl1)
  (bl0 -> bl0)

queryall (al1 + al0);(bl1 + bl0);(ar0 + ar1);(br0 + br1);(as0 + as1);(bs0 + bs1)









