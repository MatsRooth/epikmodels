define Bool ["0"|"1"];
define St0 [Bool Bool];
define MB [1 Bool];
define MA [Bool 1];
define St St0 & (MA | MB);
define Nst(X) [St - X];
define MB St & MB;
define MA St & MA;
define UnequalStPair [MB Nst(MB)] | [Nst(MB) MB] | [MA Nst(MA)] | [Nst(MA) MA];
define as1 [[[[MA as1] MA] | [[Nst(MA) as1] Nst(MA)]] & [[[MB as1] MB] | [[Nst(MB) as1] Nst(MB)]]];
define as0 [[[[MA as0] MA] | [[Nst(MA) as0] Nst(MA)]] & [[[MB as0] MB] | [[Nst(MB) as0] Nst(MB)]]];
define bs1 [[[[MA bs1] MA] | [[Nst(MA) bs1] Nst(MA)]] & [[[MB bs1] MB] | [[Nst(MB) bs1] Nst(MB)]]];
define bs0 [[[[MA bs0] MA] | [[Nst(MA) bs0] Nst(MA)]] & [[[MB bs0] MB] | [[Nst(MB) bs0] Nst(MB)]]];
define al1 [[[[MA al1] MA] | [[Nst(MA) al1] Nst(MA)]] & [[MB al1] MB]];
define al0 [[[[MA al0] MA] | [[Nst(MA) al0] Nst(MA)]] & [[Nst(MB) al0] Nst(MB)]];
define bl1 [[[MA bl1] MA] & [[[MB bl1] MB] | [[Nst(MB) bl1] Nst(MB)]]];
define bl0 [[[Nst(MA) bl0] Nst(MA)] & [[[MB bl0] MB] | [[Nst(MB) bl0] Nst(MB)]]];
define Event [as1 | as0 | bs1 | bs0 | al1 | al0 | bl1 | bl0];
source kat.fst
define amy RelKst([[as1 .o. [Event .x. Event] .o. as1] | [[as0 .o. [Event .x. Event] .o. as0] | [[al1 .o. [Event .x. Event] .o. al1] | [[al0 .o. [Event .x. Event] .o. al0] | [[bs1 .o. [Event .x. Event] .o. bs1] | [[bs0 .o. [Event .x. Event] .o. bs0] | [[bl1 .o. [Event .x. Event] .o. [bl1 | bl0]] | [bl0 .o. [Event .x. Event] .o. [bl1 | bl0]]]]]]]]]);
define bob RelKst([[as1 .o. [Event .x. Event] .o. as1] | [[as0 .o. [Event .x. Event] .o. as0] | [[al1 .o. [Event .x. Event] .o. [al1 | al0]] | [[al0 .o. [Event .x. Event] .o. [al1 | al0]] | [[bs1 .o. [Event .x. Event] .o. bs1] | [[bs0 .o. [Event .x. Event] .o. bs0] | [[bl1 .o. [Event .x. Event] .o. bl1] | [bl0 .o. [Event .x. Event] .o. bl0]]]]]]]]);
regex Cn([al1 | al0],[bl1 | bl0]);
set print-space ON
print words

