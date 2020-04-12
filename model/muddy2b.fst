define Bool ["0"|"1"];
define St0 [Bool Bool Bool Bool];
define M1 [1 Bool Bool Bool];
define M2 [Bool 1 Bool Bool];
define L1 [Bool Bool 1 Bool];
define L2 [Bool Bool Bool 1];
define St St0 & (((M1 | M2) & (L1 | L2)) & (St0 - (L1 & L2)));
define Nst(X) [St - X];
define M1 St & M1;
define M2 St & M2;
define L1 St & L1;
define L2 St & L2;
define UnequalStPair [M1 Nst(M1)] | [Nst(M1) M1] | [M2 Nst(M2)] | [Nst(M2) M2] | [L1 Nst(L1)] | [Nst(L1) L1] | [L2 Nst(L2)] | [Nst(L2) L2];
define c1%_yes [[[[L2 c1%_yes] [L1 & [St - L2]]] & [[[M1 c1%_yes] M1] | [[[St - M1] c1%_yes] [St - M1]]]] & [[[M2 c1%_yes] M2] | [[[St - M2] c1%_yes] [St - M2]]]];
define c1%_no [[[[L2 c1%_no] [L1 & [St - L2]]] & [[[M1 c1%_no] M1] | [[[St - M1] c1%_no] [St - M1]]]] & [[[M2 c1%_no] M2] | [[[St - M2] c1%_no] [St - M2]]]];
define c2%_yes [[[[L1 c2%_yes] [L2 & [St - L1]]] & [[[M1 c2%_yes] M1] | [[[St - M1] c2%_yes] [St - M1]]]] & [[[M2 c2%_yes] M2] | [[[St - M2] c2%_yes] [St - M2]]]];
define c2%_no [[[[L1 c2%_no] [L2 & [St - L1]]] & [[[M1 c2%_no] M1] | [[[St - M1] c2%_no] [St - M1]]]] & [[[M2 c2%_no] M2] | [[[St - M2] c2%_no] [St - M2]]]];
define c1%_look%_m [[[[[L2 & M2] c1%_look%_m] [L1 & [St - L2]]] & [[[M1 c1%_look%_m] M1] | [[[St - M1] c1%_look%_m] [St - M1]]]] & [[[M2 c1%_look%_m] M2] | [[[St - M2] c1%_look%_m] [St - M2]]]];
define c1%_look%_f [[[[[L2 & [St - M2]] c1%_look%_f] [L1 & [St - L2]]] & [[[M1 c1%_look%_f] M1] | [[[St - M1] c1%_look%_f] [St - M1]]]] & [[[M2 c1%_look%_f] M2] | [[[St - M2] c1%_look%_f] [St - M2]]]];
define c2%_look%_m [[[[[L1 & M1] c2%_look%_m] [L2 & [St - L1]]] & [[[M1 c2%_look%_m] M1] | [[[St - M1] c2%_look%_m] [St - M1]]]] & [[[M2 c2%_look%_m] M2] | [[[St - M2] c2%_look%_m] [St - M2]]]];
define c2%_look%_f [[[[[L1 & [St - M1]] c2%_look%_f] [L2 & [St - L1]]] & [[[M1 c2%_look%_f] M1] | [[[St - M1] c2%_look%_f] [St - M1]]]] & [[[M2 c2%_look%_f] M2] | [[[St - M2] c2%_look%_f] [St - M2]]]];
define Event [c1%_yes | c1%_no | c2%_yes | c2%_no | c1%_look%_m | c1%_look%_f | c2%_look%_m | c2%_look%_f];
source kat.fst
define c1 RelKst([[c1%_yes .o. [Event .x. Event] .o. c1%_yes] | [[c1%_no .o. [Event .x. Event] .o. c1%_no] | [[c1%_look%_m .o. [Event .x. Event] .o. c1%_look%_m] | [[c1%_look%_f .o. [Event .x. Event] .o. c1%_look%_f] | [[c2%_yes .o. [Event .x. Event] .o. c2%_yes] | [[c2%_no .o. [Event .x. Event] .o. c2%_no] | [[c2%_look%_m .o. [Event .x. Event] .o. [c2%_look%_m | c2%_look%_f]] | [c2%_look%_f .o. [Event .x. Event] .o. [c2%_look%_m | c2%_look%_f]]]]]]]]]);
define c2 RelKst([[c1%_yes .o. [Event .x. Event] .o. c1%_yes] | [[c1%_no .o. [Event .x. Event] .o. c1%_no] | [[c1%_look%_m .o. [Event .x. Event] .o. [c1%_look%_m | c1%_look%_f]] | [[c1%_look%_f .o. [Event .x. Event] .o. [c1%_look%_m | c1%_look%_f]] | [[c2%_yes .o. [Event .x. Event] .o. c2%_yes] | [[c2%_no .o. [Event .x. Event] .o. c2%_no] | [[c2%_look%_m .o. [Event .x. Event] .o. c2%_look%_m] | [c2%_look%_f .o. [Event .x. Event] .o. c2%_look%_f]]]]]]]]);
regex Cn(L2,Cn([c1%_look%_m | c1%_look%_f],[c2%_look%_m | c2%_look%_f]));
set print-space ON
print words

