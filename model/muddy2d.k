world = MA+MB+WA+WB

assert (MA+MB)

action amy_rfl_1 = (((test WA)+(~test WA));id;(test WA)) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))
action amy_rfl_O = (((test WA)+(~test WA));id;(~test WA)) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))

action bob_rfl_1 = (((test WB)+(~test WB));id;(test WB)) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))
action bob_rfl_O = (((test WB)+(~test WB));id;(~test WB)) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))

action amy_say_1 = ((test WA);id;(test WA)) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))
action amy_say_O = ((~test WA);id;(~test WA)) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))

action bob_say_1 = ((test WB);id;(test WB)) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))
action bob_say_O = ((~test WB);id;(~test WB)) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB))))

action amy_look_m = ((test MB);id;(test MB)) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB)))) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB))))
action amy_look_f = ((~test MB);id;(~test MB)) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB)))) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB))))

action bob_look_m = ((test MA);id;(test MA)) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB)))) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB))))
action bob_look_f = ((~test MA);id;(~test MA)) & (((test MA);id;(test MA))+((~(test MA));id;(~(test MA)))) & (((test MB);id;(test MB))+((~(test MB));id;(~(test MB)))) & (((test WA);id;(test WA))+((~(test WA));id;(~(test WA)))) & (((test WB);id;(test WB))+((~(test WB));id;(~(test WB))))


agent amy =
  (amy_say_1 -> amy_say_1)
  (amy_say_O -> amy_say_O)
  (amy_rfl_1 -> amy_rfl_1)
  (amy_rfl_O -> amy_rfl_O)
  (amy_look_m -> amy_look_m)
  (amy_look_f -> amy_look_f)
  (bob_say_1 -> bob_say_1)
  (bob_say_O -> bob_say_O)
  (bob_rfl_1 -> bob_rfl_O + bob_rfl_1)
  (bob_rfl_O -> bob_rfl_O + bob_rfl_1)
  (bob_look_m -> bob_look_m + bob_look_f)
  (bob_look_f -> bob_look_m + bob_look_f)


agent bob =
  (amy_say_1 -> amy_say_1)
  (amy_say_O -> amy_say_O)
  (amy_rfl_1 -> amy_rfl_O + amy_rfl_1)
  (amy_rfl_O -> amy_rfl_O + amy_rfl_1)
  (amy_look_m -> amy_look_m + amy_look_f)
  (amy_look_f -> amy_look_m + amy_look_f)
  (bob_say_1 -> bob_say_1)
  (bob_say_O -> bob_say_O)
  (bob_rfl_1 -> bob_rfl_1)
  (bob_rfl_O -> bob_rfl_O)
  (bob_look_m -> bob_look_m)
  (bob_look_f -> bob_look_f)

queryall (amy_look_m + amy_look_f);(bob_look_m + bob_look_f);(amy_rfl_O + amy_rfl_1);(bob_rfl_O + bob_rfl_1);(amy_say_O + amy_say_1);(bob_say_O + bob_say_1)









