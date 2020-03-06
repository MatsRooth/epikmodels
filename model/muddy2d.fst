define Bool ["0"|"1"];
define St0 [Bool Bool Bool Bool];
define MA [1 Bool Bool Bool];
define MB [Bool 1 Bool Bool];
define WA [Bool Bool 1 Bool];
define WB [Bool Bool Bool 1];
define St St0 & (MA | MB);
define Nst(X) [St - X];
define MA St & MA;
define MB St & MB;
define WA St & WA;
define WB St & WB;
define UnequalStPair [MA Nst(MA)] | [Nst(MA) MA] | [MB Nst(MB)] | [Nst(MB) MB] | [WA Nst(WA)] | [Nst(WA) WA] | [WB Nst(WB)] | [Nst(WB) WB];
define amy%_rfl%_1 [[[[[[WA | [St - WA]] amy%_rfl%_1] WA] & [[[WB amy%_rfl%_1] WB] | [[[St - WB] amy%_rfl%_1] [St - WB]]]] & [[[MA amy%_rfl%_1] MA] | [[[St - MA] amy%_rfl%_1] [St - MA]]]] & [[[MB amy%_rfl%_1] MB] | [[[St - MB] amy%_rfl%_1] [St - MB]]]];
define amy%_rfl%_O [[[[[[WA | [St - WA]] amy%_rfl%_O] [St - WA]] & [[[WB amy%_rfl%_O] WB] | [[[St - WB] amy%_rfl%_O] [St - WB]]]] & [[[MA amy%_rfl%_O] MA] | [[[St - MA] amy%_rfl%_O] [St - MA]]]] & [[[MB amy%_rfl%_O] MB] | [[[St - MB] amy%_rfl%_O] [St - MB]]]];
define bob%_rfl%_1 [[[[[[WB | [St - WB]] bob%_rfl%_1] WB] & [[[WA bob%_rfl%_1] WA] | [[[St - WA] bob%_rfl%_1] [St - WA]]]] & [[[MA bob%_rfl%_1] MA] | [[[St - MA] bob%_rfl%_1] [St - MA]]]] & [[[MB bob%_rfl%_1] MB] | [[[St - MB] bob%_rfl%_1] [St - MB]]]];
define bob%_rfl%_O [[[[[[WB | [St - WB]] bob%_rfl%_O] [St - WB]] & [[[WA bob%_rfl%_O] WA] | [[[St - WA] bob%_rfl%_O] [St - WA]]]] & [[[MA bob%_rfl%_O] MA] | [[[St - MA] bob%_rfl%_O] [St - MA]]]] & [[[MB bob%_rfl%_O] MB] | [[[St - MB] bob%_rfl%_O] [St - MB]]]];
define amy%_say%_1 [[[[[WA amy%_say%_1] WA] & [[[WB amy%_say%_1] WB] | [[[St - WB] amy%_say%_1] [St - WB]]]] & [[[MA amy%_say%_1] MA] | [[[St - MA] amy%_say%_1] [St - MA]]]] & [[[MB amy%_say%_1] MB] | [[[St - MB] amy%_say%_1] [St - MB]]]];
define amy%_say%_O [[[[[[St - WA] amy%_say%_O] [St - WA]] & [[[WB amy%_say%_O] WB] | [[[St - WB] amy%_say%_O] [St - WB]]]] & [[[MA amy%_say%_O] MA] | [[[St - MA] amy%_say%_O] [St - MA]]]] & [[[MB amy%_say%_O] MB] | [[[St - MB] amy%_say%_O] [St - MB]]]];
define bob%_say%_1 [[[[[WB bob%_say%_1] WB] & [[[WA bob%_say%_1] WA] | [[[St - WA] bob%_say%_1] [St - WA]]]] & [[[MA bob%_say%_1] MA] | [[[St - MA] bob%_say%_1] [St - MA]]]] & [[[MB bob%_say%_1] MB] | [[[St - MB] bob%_say%_1] [St - MB]]]];
define bob%_say%_O [[[[[[St - WB] bob%_say%_O] [St - WB]] & [[[WA bob%_say%_O] WA] | [[[St - WA] bob%_say%_O] [St - WA]]]] & [[[MA bob%_say%_O] MA] | [[[St - MA] bob%_say%_O] [St - MA]]]] & [[[MB bob%_say%_O] MB] | [[[St - MB] bob%_say%_O] [St - MB]]]];
define amy%_look%_m [[[[[[MB amy%_look%_m] MB] & [[[MA amy%_look%_m] MA] | [[[St - MA] amy%_look%_m] [St - MA]]]] & [[[MB amy%_look%_m] MB] | [[[St - MB] amy%_look%_m] [St - MB]]]] & [[[WA amy%_look%_m] WA] | [[[St - WA] amy%_look%_m] [St - WA]]]] & [[[WB amy%_look%_m] WB] | [[[St - WB] amy%_look%_m] [St - WB]]]];
define amy%_look%_f [[[[[[[St - MB] amy%_look%_f] [St - MB]] & [[[MA amy%_look%_f] MA] | [[[St - MA] amy%_look%_f] [St - MA]]]] & [[[MB amy%_look%_f] MB] | [[[St - MB] amy%_look%_f] [St - MB]]]] & [[[WA amy%_look%_f] WA] | [[[St - WA] amy%_look%_f] [St - WA]]]] & [[[WB amy%_look%_f] WB] | [[[St - WB] amy%_look%_f] [St - WB]]]];
define bob%_look%_m [[[[[[MA bob%_look%_m] MA] & [[[MA bob%_look%_m] MA] | [[[St - MA] bob%_look%_m] [St - MA]]]] & [[[MB bob%_look%_m] MB] | [[[St - MB] bob%_look%_m] [St - MB]]]] & [[[WA bob%_look%_m] WA] | [[[St - WA] bob%_look%_m] [St - WA]]]] & [[[WB bob%_look%_m] WB] | [[[St - WB] bob%_look%_m] [St - WB]]]];
define bob%_look%_f [[[[[[[St - MA] bob%_look%_f] [St - MA]] & [[[MA bob%_look%_f] MA] | [[[St - MA] bob%_look%_f] [St - MA]]]] & [[[MB bob%_look%_f] MB] | [[[St - MB] bob%_look%_f] [St - MB]]]] & [[[WA bob%_look%_f] WA] | [[[St - WA] bob%_look%_f] [St - WA]]]] & [[[WB bob%_look%_f] WB] | [[[St - WB] bob%_look%_f] [St - WB]]]];
define Event [amy%_rfl%_1 | amy%_rfl%_O | bob%_rfl%_1 | bob%_rfl%_O | amy%_say%_1 | amy%_say%_O | bob%_say%_1 | bob%_say%_O | amy%_look%_m | amy%_look%_f | bob%_look%_m | bob%_look%_f];
source kat.fst
define amy RelKst([[amy%_say%_1 .o. [Event .x. Event] .o. amy%_say%_1] | [[amy%_say%_O .o. [Event .x. Event] .o. amy%_say%_O] | [[amy%_rfl%_1 .o. [Event .x. Event] .o. amy%_rfl%_1] | [[amy%_rfl%_O .o. [Event .x. Event] .o. amy%_rfl%_O] | [[amy%_look%_m .o. [Event .x. Event] .o. amy%_look%_m] | [[amy%_look%_f .o. [Event .x. Event] .o. amy%_look%_f] | [[bob%_say%_1 .o. [Event .x. Event] .o. bob%_say%_1] | [[bob%_say%_O .o. [Event .x. Event] .o. bob%_say%_O] | [[bob%_rfl%_1 .o. [Event .x. Event] .o. [bob%_rfl%_O | bob%_rfl%_1]] | [[bob%_rfl%_O .o. [Event .x. Event] .o. [bob%_rfl%_O | bob%_rfl%_1]] | [[bob%_look%_m .o. [Event .x. Event] .o. [bob%_look%_m | bob%_look%_f]] | [bob%_look%_f .o. [Event .x. Event] .o. [bob%_look%_m | bob%_look%_f]]]]]]]]]]]]]);
define bob RelKst([[amy%_say%_1 .o. [Event .x. Event] .o. amy%_say%_1] | [[amy%_say%_O .o. [Event .x. Event] .o. amy%_say%_O] | [[amy%_rfl%_1 .o. [Event .x. Event] .o. [amy%_rfl%_O | amy%_rfl%_1]] | [[amy%_rfl%_O .o. [Event .x. Event] .o. [amy%_rfl%_O | amy%_rfl%_1]] | [[amy%_look%_m .o. [Event .x. Event] .o. [amy%_look%_m | amy%_look%_f]] | [[amy%_look%_f .o. [Event .x. Event] .o. [amy%_look%_m | amy%_look%_f]] | [[bob%_say%_1 .o. [Event .x. Event] .o. bob%_say%_1] | [[bob%_say%_O .o. [Event .x. Event] .o. bob%_say%_O] | [[bob%_rfl%_1 .o. [Event .x. Event] .o. bob%_rfl%_1] | [[bob%_rfl%_O .o. [Event .x. Event] .o. bob%_rfl%_O] | [[bob%_look%_m .o. [Event .x. Event] .o. bob%_look%_m] | [bob%_look%_f .o. [Event .x. Event] .o. bob%_look%_f]]]]]]]]]]]]);
regex Cn([amy%_look%_m | amy%_look%_f],Cn([bob%_look%_m | bob%_look%_f],Cn([amy%_rfl%_O | amy%_rfl%_1],Cn([bob%_rfl%_O | bob%_rfl%_1],Cn([amy%_say%_O | amy%_say%_1],[bob%_say%_O | bob%_say%_1])))));
set print-space ON
print words

