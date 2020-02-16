define Bool ["0"|"1"];
define St0 [Bool Bool Bool Bool];
define H1 [1 Bool Bool Bool];
define T1 [Bool 1 Bool Bool];
define H2 [Bool Bool 1 Bool];
define T2 [Bool Bool Bool 1];
define St St0 & ((((St0 - (H1 & T1)) & (H1 | T1)) & (St0 - (H2 & T2))) & (H2 | T2));
define Nst(X) [St - X];
define H1 St & H1;
define T1 St & T1;
define H2 St & H2;
define T2 St & T2;
define UnequalStPair [H1 Nst(H1)] | [Nst(H1) H1] | [T1 Nst(T1)] | [Nst(T1) T1] | [H2 Nst(H2)] | [Nst(H2) H2] | [T2 Nst(T2)] | [Nst(T2) T2];
define announce%_H1 [[[H1 announce%_H1] H1] & [[[H2 announce%_H1] H2] | [[T2 announce%_H1] T2]]];
define announce%_T1 [[[T1 announce%_T1] T1] & [[[H2 announce%_T1] H2] | [[T2 announce%_T1] T2]]];
define announce%_H2 [[[H2 announce%_H2] H2] & [[[H1 announce%_H2] H1] | [[T1 announce%_H2] T1]]];
define announce%_T2 [[[T2 announce%_T2] T2] & [[[H1 announce%_T2] H1] | [[T1 announce%_T2] T1]]];
define peek%_amy%_H1 [[[H1 peek%_amy%_H1] H1] & [[[H2 peek%_amy%_H1] H2] | [[T2 peek%_amy%_H1] T2]]];
define peek%_amy%_T1 [[[T1 peek%_amy%_T1] T1] & [[[H2 peek%_amy%_T1] H2] | [[T2 peek%_amy%_T1] T2]]];
define peek%_amy%_H2 [[[H2 peek%_amy%_H2] H2] & [[[H1 peek%_amy%_H2] H1] | [[T1 peek%_amy%_H2] T1]]];
define peek%_amy%_T2 [[[T2 peek%_amy%_T2] T2] & [[[H1 peek%_amy%_T2] H1] | [[T1 peek%_amy%_T2] T1]]];
define peek%_bob%_H1 [[[H1 peek%_bob%_H1] H1] & [[[H2 peek%_bob%_H1] H2] | [[T2 peek%_bob%_H1] T2]]];
define peek%_bob%_T1 [[[T1 peek%_bob%_T1] T1] & [[[H2 peek%_bob%_T1] H2] | [[T2 peek%_bob%_T1] T2]]];
define peek%_bob%_H2 [[[H2 peek%_bob%_H2] H2] & [[[H1 peek%_bob%_H2] H1] | [[T1 peek%_bob%_H2] T1]]];
define peek%_bob%_T2 [[[T2 peek%_bob%_T2] T2] & [[[H1 peek%_bob%_T2] H1] | [[T1 peek%_bob%_T2] T1]]];
define Event [announce%_H1 | announce%_T1 | announce%_H2 | announce%_T2 | peek%_amy%_H1 | peek%_amy%_T1 | peek%_amy%_H2 | peek%_amy%_T2 | peek%_bob%_H1 | peek%_bob%_T1 | peek%_bob%_H2 | peek%_bob%_T2];
source kat.fst
define amy RelKst([[announce%_H1 .o. [Event .x. Event] .o. announce%_H1] | [[announce%_T1 .o. [Event .x. Event] .o. announce%_T1] | [[announce%_H2 .o. [Event .x. Event] .o. announce%_H2] | [[announce%_T2 .o. [Event .x. Event] .o. announce%_T2] | [[peek%_amy%_H1 .o. [Event .x. Event] .o. peek%_amy%_H1] | [[peek%_amy%_T1 .o. [Event .x. Event] .o. peek%_amy%_T1] | [[peek%_amy%_H2 .o. [Event .x. Event] .o. peek%_amy%_H2] | [[peek%_amy%_T2 .o. [Event .x. Event] .o. peek%_amy%_T2] | [[peek%_bob%_H1 .o. [Event .x. Event] .o. [peek%_bob%_H1 | peek%_bob%_T1]] | [[peek%_bob%_T1 .o. [Event .x. Event] .o. [peek%_bob%_H1 | peek%_bob%_T1]] | [[peek%_bob%_H2 .o. [Event .x. Event] .o. [peek%_bob%_H2 | peek%_bob%_T2]] | [peek%_bob%_T2 .o. [Event .x. Event] .o. [peek%_bob%_H2 | peek%_bob%_T2]]]]]]]]]]]]]);
define bob RelKst([[announce%_H1 .o. [Event .x. Event] .o. announce%_H1] | [[announce%_T1 .o. [Event .x. Event] .o. announce%_T1] | [[announce%_H2 .o. [Event .x. Event] .o. announce%_H2] | [[announce%_T2 .o. [Event .x. Event] .o. announce%_T2] | [[peek%_amy%_H1 .o. [Event .x. Event] .o. [peek%_amy%_H1 | peek%_amy%_T1]] | [[peek%_amy%_T1 .o. [Event .x. Event] .o. [peek%_amy%_H1 | peek%_amy%_T1]] | [[peek%_amy%_H2 .o. [Event .x. Event] .o. [peek%_amy%_H2 | peek%_amy%_T2]] | [[peek%_amy%_T2 .o. [Event .x. Event] .o. [peek%_amy%_H2 | peek%_amy%_T2]] | [[peek%_bob%_H1 .o. [Event .x. Event] .o. peek%_bob%_H1] | [[peek%_bob%_T1 .o. [Event .x. Event] .o. peek%_bob%_T1] | [[peek%_bob%_H2 .o. [Event .x. Event] .o. peek%_bob%_H2] | [peek%_bob%_T2 .o. [Event .x. Event] .o. peek%_bob%_T2]]]]]]]]]]]]);
regex Dia(bob,[peek%_bob%_T2 peek%_amy%_T2]);
print random-words

