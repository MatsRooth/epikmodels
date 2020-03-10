define Bool ["0"|"1"];
define St0 [Bool Bool];
define MB [1 Bool];
define MA [Bool 1];
define St St0 & (MA | MB);
define Nst(X) [St - X];
define MB St & MB;
define MA St & MA;
define UnequalStPair [MB Nst(MB)] | [Nst(MB) MB] | [MA Nst(MA)] | [Nst(MA) MA];
define amy%_say%_1 [[[[MA amy%_say%_1] MA] | [[[St - MA] amy%_say%_1] [St - MA]]] & [[[MB amy%_say%_1] MB] | [[[St - MB] amy%_say%_1] [St - MB]]]];
define amy%_say%_O [[[[MA amy%_say%_O] MA] | [[[St - MA] amy%_say%_O] [St - MA]]] & [[[MB amy%_say%_O] MB] | [[[St - MB] amy%_say%_O] [St - MB]]]];
define bob%_say%_1 [[[[MA bob%_say%_1] MA] | [[[St - MA] bob%_say%_1] [St - MA]]] & [[[MB bob%_say%_1] MB] | [[[St - MB] bob%_say%_1] [St - MB]]]];
define bob%_say%_O [[[[MA bob%_say%_O] MA] | [[[St - MA] bob%_say%_O] [St - MA]]] & [[[MB bob%_say%_O] MB] | [[[St - MB] bob%_say%_O] [St - MB]]]];
define amy%_loo%_1 [[[[MA amy%_loo%_1] MA] | [[[St - MA] amy%_loo%_1] [St - MA]]] & [[MB amy%_loo%_1] MB]];
define amy%_loo%_O [[[[MA amy%_loo%_O] MA] | [[[St - MA] amy%_loo%_O] [St - MA]]] & [[[St - MB] amy%_loo%_O] [St - MB]]];
define bob%_loo%_1 [[[MA bob%_loo%_1] MA] & [[[MB bob%_loo%_1] MB] | [[[St - MB] bob%_loo%_1] [St - MB]]]];
define bob%_loo%_O [[[[St - MA] bob%_loo%_O] [St - MA]] & [[[MB bob%_loo%_O] MB] | [[[St - MB] bob%_loo%_O] [St - MB]]]];
define Event [amy%_say%_1 | amy%_say%_O | bob%_say%_1 | bob%_say%_O | amy%_loo%_1 | amy%_loo%_O | bob%_loo%_1 | bob%_loo%_O];
source kat.fst
define amy RelKst([[amy%_say%_1 .o. [Event .x. Event] .o. amy%_say%_1] | [[amy%_say%_O .o. [Event .x. Event] .o. amy%_say%_O] | [[amy%_loo%_1 .o. [Event .x. Event] .o. amy%_loo%_1] | [[amy%_loo%_O .o. [Event .x. Event] .o. amy%_loo%_O] | [[bob%_say%_1 .o. [Event .x. Event] .o. bob%_say%_1] | [[bob%_say%_O .o. [Event .x. Event] .o. bob%_say%_O] | [[bob%_loo%_1 .o. [Event .x. Event] .o. [bob%_loo%_1 | bob%_loo%_O]] | [bob%_loo%_O .o. [Event .x. Event] .o. [bob%_loo%_1 | bob%_loo%_O]]]]]]]]]);
define bob RelKst([[amy%_say%_1 .o. [Event .x. Event] .o. amy%_say%_1] | [[amy%_say%_O .o. [Event .x. Event] .o. amy%_say%_O] | [[amy%_loo%_1 .o. [Event .x. Event] .o. [amy%_loo%_1 | amy%_loo%_O]] | [[amy%_loo%_O .o. [Event .x. Event] .o. [amy%_loo%_1 | amy%_loo%_O]] | [[bob%_say%_1 .o. [Event .x. Event] .o. bob%_say%_1] | [[bob%_say%_O .o. [Event .x. Event] .o. bob%_say%_O] | [[bob%_loo%_1 .o. [Event .x. Event] .o. bob%_loo%_1] | [bob%_loo%_O .o. [Event .x. Event] .o. bob%_loo%_O]]]]]]]]);
regex Cn([amy%_loo%_1 | amy%_loo%_O],[bob%_loo%_1 | bob%_loo%_O]);
set print-space ON
print words

