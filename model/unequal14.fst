define Bool ["0"|"1"];
define St14 [Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool];

define B11 ["1" Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool];
define B10 ["0" Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool];

define B21 [Bool "1" Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool];
define B20 [Bool "0" Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool];

define B31 [Bool Bool "1" Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool];
define B30 [Bool Bool "0" Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool];

define B41 [Bool Bool Bool "1" Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool];
define B40 [Bool Bool Bool "0" Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool];

define B51 [Bool Bool Bool Bool "1" Bool Bool Bool Bool Bool Bool Bool Bool Bool];
define B50 [Bool Bool Bool Bool "0" Bool Bool Bool Bool Bool Bool Bool Bool Bool];

define B61 [Bool Bool Bool Bool Bool "1" Bool Bool Bool Bool Bool Bool Bool Bool];
define B60 [Bool Bool Bool Bool Bool "0" Bool Bool Bool Bool Bool Bool Bool Bool];

define B71 [Bool Bool Bool Bool Bool Bool "1" Bool Bool Bool Bool Bool Bool Bool];
define B70 [Bool Bool Bool Bool Bool Bool "0" Bool Bool Bool Bool Bool Bool Bool];

define B81 [Bool Bool Bool Bool Bool Bool Bool "1" Bool Bool Bool Bool Bool Bool];
define B80 [Bool Bool Bool Bool Bool Bool Bool "0" Bool Bool Bool Bool Bool Bool];

define B91 [Bool Bool Bool Bool Bool Bool Bool Bool "1" Bool Bool Bool Bool Bool];
define B90 [Bool Bool Bool Bool Bool Bool Bool Bool "0" Bool Bool Bool Bool Bool];

define B101 [Bool Bool Bool Bool Bool Bool Bool Bool Bool "1" Bool Bool Bool Bool];
define B100 [Bool Bool Bool Bool Bool Bool Bool Bool Bool "0" Bool Bool Bool Bool];

define B111 [Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool "1" Bool Bool Bool];
define B110 [Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool "0" Bool Bool Bool];

define B121 [Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool "1" Bool Bool];
define B120 [Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool "0" Bool Bool];

define B131 [Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool "1" Bool];
define B130 [Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool "0" Bool];

define B141 [Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool "1"];
define B140 [Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool Bool "0"];

define UnequalStPair [B11 B10] | [B10 B11] |
[B21 B20] | [B20 B21] |
[B31 B30] | [B30 B31] |
[B41 B40] | [B40 B41] |
[B51 B50] | [B50 B51] |
[B61 B60] | [B60 B61] |
[B71 B70] | [B70 B71] |
[B81 B80] | [B80 B81] |
[B91 B90] | [B90 B91] |
[B101 B100] | [B100 B101] |
[B111 B110] | [B110 B111] |
[B121 B120] | [B120 B121] |
[B131 B130] | [B130 B131] |
[B141 B140] | [B140 B141]; 

define Wf0 ~[$ UnequalStPair];

# Reduce a sequence of two generators to a single one, by deleting the second one.
define Squash St14 -> 0 || St14 _;


undefine Bool St14
undefine  B11 B10
undefine  B21 B20
undefine  B31 B30
undefine  B41 B40
undefine  B51 B50
undefine  B61 B60
undefine  B71 B70
undefine  B81 B80
undefine  B91 B90
undefine  B101 B100 
undefine  B111 B110 
undefine  B121 B120 
undefine  B131 B130
undefine  B141 B140 


save defined unequal14.net

# 'UnequalStPair': 2.3 Mb. 28699 states, 57394 arcs, 268414976 paths.
# 'Wf0': 7.3 Mb. 70360 states, 210851 arcs, Circular.

