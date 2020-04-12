define Bool ["0"|"1"];
define St0 [Bool Bool Bool Bool];
define MB [1 Bool Bool Bool];
define MA [Bool 1 Bool Bool];
define WB [Bool Bool 1 Bool];
define WA [Bool Bool Bool 1];
define St St0 & (MA | MB);
define Nst(X) [St - X];
define MB St & MB;
define MA St & MA;
define WB St & WB;
define WA St & WA;
define UnequalStPair [MB Nst(MB)] | [Nst(MB) MB] | [MA Nst(MA)] | [Nst(MA) MA] | [WB Nst(WB)] | [Nst(WB) WB] | [WA Nst(WA)] | [Nst(WA) WA];
define ar1 [[[[[[WA | Nst(WA)] ar1] WA] & [[[WB ar1] WB] | [[Nst(WB) ar1] Nst(WB)]]] & [[[MA ar1] MA] | [[Nst(MA) ar1] Nst(MA)]]] & [[[MB ar1] MB] | [[Nst(MB) ar1] Nst(MB)]]];
define ar0 [[[[[[WA | Nst(WA)] ar0] Nst(WA)] & [[[WB ar0] WB] | [[Nst(WB) ar0] Nst(WB)]]] & [[[MA ar0] MA] | [[Nst(MA) ar0] Nst(MA)]]] & [[[MB ar0] MB] | [[Nst(MB) ar0] Nst(MB)]]];
define br1 [[[[[[WB | Nst(WB)] br1] WB] & [[[WA br1] WA] | [[Nst(WA) br1] Nst(WA)]]] & [[[MA br1] MA] | [[Nst(MA) br1] Nst(MA)]]] & [[[MB br1] MB] | [[Nst(MB) br1] Nst(MB)]]];
define br0 [[[[[[WB | Nst(WB)] br0] Nst(WB)] & [[[WA br0] WA] | [[Nst(WA) br0] Nst(WA)]]] & [[[MA br0] MA] | [[Nst(MA) br0] Nst(MA)]]] & [[[MB br0] MB] | [[Nst(MB) br0] Nst(MB)]]];
define as1 [[[[[WA as1] WA] & [[[WB as1] WB] | [[Nst(WB) as1] Nst(WB)]]] & [[[MA as1] MA] | [[Nst(MA) as1] Nst(MA)]]] & [[[MB as1] MB] | [[Nst(MB) as1] Nst(MB)]]];
define as0 [[[[[Nst(WA) as0] Nst(WA)] & [[[WB as0] WB] | [[Nst(WB) as0] Nst(WB)]]] & [[[MA as0] MA] | [[Nst(MA) as0] Nst(MA)]]] & [[[MB as0] MB] | [[Nst(MB) as0] Nst(MB)]]];
define bs1 [[[[[WB bs1] WB] & [[[WA bs1] WA] | [[Nst(WA) bs1] Nst(WA)]]] & [[[MA bs1] MA] | [[Nst(MA) bs1] Nst(MA)]]] & [[[MB bs1] MB] | [[Nst(MB) bs1] Nst(MB)]]];
define bs0 [[[[[Nst(WB) bs0] Nst(WB)] & [[[WA bs0] WA] | [[Nst(WA) bs0] Nst(WA)]]] & [[[MA bs0] MA] | [[Nst(MA) bs0] Nst(MA)]]] & [[[MB bs0] MB] | [[Nst(MB) bs0] Nst(MB)]]];
define al1 [[[[[[MB al1] MB] & [[[MA al1] MA] | [[Nst(MA) al1] Nst(MA)]]] & [[[MB al1] MB] | [[Nst(MB) al1] Nst(MB)]]] & [[[WA al1] WA] | [[Nst(WA) al1] Nst(WA)]]] & [[[WB al1] WB] | [[Nst(WB) al1] Nst(WB)]]];
define al0 [[[[[[Nst(MB) al0] Nst(MB)] & [[[MA al0] MA] | [[Nst(MA) al0] Nst(MA)]]] & [[[MB al0] MB] | [[Nst(MB) al0] Nst(MB)]]] & [[[WA al0] WA] | [[Nst(WA) al0] Nst(WA)]]] & [[[WB al0] WB] | [[Nst(WB) al0] Nst(WB)]]];
define bl1 [[[[[[MA bl1] MA] & [[[MA bl1] MA] | [[Nst(MA) bl1] Nst(MA)]]] & [[[MB bl1] MB] | [[Nst(MB) bl1] Nst(MB)]]] & [[[WA bl1] WA] | [[Nst(WA) bl1] Nst(WA)]]] & [[[WB bl1] WB] | [[Nst(WB) bl1] Nst(WB)]]];
define bl0 [[[[[[Nst(MA) bl0] Nst(MA)] & [[[MA bl0] MA] | [[Nst(MA) bl0] Nst(MA)]]] & [[[MB bl0] MB] | [[Nst(MB) bl0] Nst(MB)]]] & [[[WA bl0] WA] | [[Nst(WA) bl0] Nst(WA)]]] & [[[WB bl0] WB] | [[Nst(WB) bl0] Nst(WB)]]];
define Event [ar1 | ar0 | br1 | br0 | as1 | as0 | bs1 | bs0 | al1 | al0 | bl1 | bl0];
source kat.fst
define amy RelKst([[as1 .o. [Event .x. Event] .o. as1] | [[as0 .o. [Event .x. Event] .o. as0] | [[ar1 .o. [Event .x. Event] .o. ar1] | [[ar0 .o. [Event .x. Event] .o. ar0] | [[al1 .o. [Event .x. Event] .o. al1] | [[al0 .o. [Event .x. Event] .o. al0] | [[bs1 .o. [Event .x. Event] .o. bs1] | [[bs0 .o. [Event .x. Event] .o. bs0] | [[br1 .o. [Event .x. Event] .o. [br0 | br1]] | [[br0 .o. [Event .x. Event] .o. [br0 | br1]] | [[bl1 .o. [Event .x. Event] .o. [bl1 | bl0]] | [bl0 .o. [Event .x. Event] .o. [bl1 | bl0]]]]]]]]]]]]]);
define bob RelKst([[as1 .o. [Event .x. Event] .o. as1] | [[as0 .o. [Event .x. Event] .o. as0] | [[ar1 .o. [Event .x. Event] .o. [ar0 | ar1]] | [[ar0 .o. [Event .x. Event] .o. [ar0 | ar1]] | [[al1 .o. [Event .x. Event] .o. [al1 | al0]] | [[al0 .o. [Event .x. Event] .o. [al1 | al0]] | [[bs1 .o. [Event .x. Event] .o. bs1] | [[bs0 .o. [Event .x. Event] .o. bs0] | [[br1 .o. [Event .x. Event] .o. br1] | [[br0 .o. [Event .x. Event] .o. br0] | [[bl1 .o. [Event .x. Event] .o. bl1] | [bl0 .o. [Event .x. Event] .o. bl0]]]]]]]]]]]]);
regex Cn([al1 | al0],Cn([bl1 | bl0],Cn([ar0 | ar1],Cn([br0 | br1],Cn([as0 | as1],[bs0 | bs1])))));
set print-space ON
print words

