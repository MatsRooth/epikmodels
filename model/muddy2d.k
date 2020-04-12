world = MB+MA+WB+WA

assert (MA+MB)

action ar1 = (((test WA)+(~test WA));id;(test WA)) &
             (constant WB) &
	     (constant MA) & (constant MB)

action ar0 = (((test WA)+(~test WA));id;(~test WA)) &
             (constant WA) &
	     (constant MA) & (constant MB)


action br1 = (((test WB)+(~test WB));id;(test WB)) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))
action br0 = (((test WB)+(~test WB));id;(~test WB)) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))

action as1 = ((test WA);id;(test WA)) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))
action as0 = ((~test WA);id;(~test WA)) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))

action bs1 = ((test WB);id;(test WB)) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))
action bs0 = ((~test WB);id;(~test WB)) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))

action al1 = ((test MB);id;(test MB)) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB)))) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB))))
action al0 = ((~test MB);id;(~test MB)) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB)))) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB))))

action bl1 = ((test MA);id;(test MA)) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB)))) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB))))
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









