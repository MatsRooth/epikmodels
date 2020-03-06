source muddy2d.fst
source whether.fst

define amylook [amy%_look%_m | amy%_look%_f];
define boblook [bob%_look%_m | bob%_look%_f];
define amyreflect [amy%_rfl%_O | amy%_rfl%_1];
define bobreflect [bob%_rfl%_O | bob%_rfl%_1];
define amysay [amy%_say%_O | amy%_say%_1];
define bobsay [bob%_say%_O | bob%_say%_1];

define W0 Nst(WA) & Nst(WB);
# 0 1 0 0
# 1 0 0 0
# 1 1 0 0

define amy0 W0 .o. amy .o. W0;
define bob0 W0 .o. bob .o. W0;

define W1 Cn(W0,amylook);
# 1 1 0 0 amy_look_m 1 1 0 0
# 1 0 0 0 amy_look_f 1 0 0 0
# 0 1 0 0 amy_look_m 0 1 0 0

define amy1 W1 .o. amy .o. W1;
define bob1 W1 .o. bob .o. W1;

define W2 Cn(W1,boblook);
# 1 1 0 0 amy_look_m 1 1 0 0 bob_look_m 1 1 0 0
# 1 0 0 0 amy_look_f 1 0 0 0 bob_look_m 1 0 0 0
# 0 1 0 0 amy_look_m 0 1 0 0 bob_look_f 0 1 0 0

define amy2 W2 .o. amy .o. W2;
define bob2 W2 .o. bob .o. W2;

# Whether Amy and Bob look at mud or flesh is determined by the state.
# There is no change yet in the whether bits.

define dontknow(W,R,F) W & [Cnr(R,[St .x. F]).u] & [Cnr(R,[St .x. Nst(F)]).u];

define doknow(W,R,F) W - dontknow(W,R,F);

define testdo(W,R,F,A1,A2) Cn(doknow(W,R,F),A1) | Cn(dontknow(W,R,F),A2); 

# regex testdo(W2,amy2,MA,amy%_rfl%_1,amy%_rfl%_O);

# Amy reflects. This should set her whether bit.

define W3 testdo(W2,amy2,MA,amy%_rfl%_1,amy%_rfl%_O);
define amy3 W3 .o. amy .o. W3;
define bob3 W3 .o. bob .o. W3;

# Bob reflects. In W4, Amy whether bit (the third bit WA), and Bob's (the fourth
# bit WB) are set as desired.
define W4 testdo(W3,bob3,MB,bob%_rfl%_1,bob%_rfl%_O);
# 1 1 0 0 amy_look_m 1 1 0 0 bob_look_m 1 1 0 0 amy_rfl_O 1 1 0 0 bob_rfl_O 1 1 0 0
# 1 0 0 0 amy_look_f 1 0 0 0 bob_look_m 1 0 0 0 amy_rfl_1 1 0 1 0 bob_rfl_O 1 0 1 0
# 0 1 0 0 amy_look_m 0 1 0 0 bob_look_f 0 1 0 0 amy_rfl_O 0 1 0 0 bob_rfl_1 0 1 0 1

define amy4 W4 .o. amy .o. W4;
define bob4 W4 .o. bob .o. W4;

# Amy says yes or no. This is a concatenative update, and her behavior is deterministic,
# since the preconditions of amy_say_O and amy_say_1 are MA and its complement, respectively.

define W5 Cn(W4,amysay);
define amy5 W5 .o. amy .o. W5;
define bob5 W5 .o. bob .o. W5;

# Bob says yes or no, in a concatenative update. (Should this be called an update?)
define W6 Cn(W5,bobsay);
define amy6 W6 .o. amy .o. W6;
define bob6 W6 .o. bob .o. W6;

regex W6;
print words

# Amy reflects. This is expressed with a test update isomorphic to the one that
# produced W3.
# Reflecting lets her draw conclusions from what Bob said.
define W7 testdo(W6,amy7,MA,amy%_rfl%_1,amy%_rfl%_O);


# Amy has identified here world in each world of W7. This is verified by subtracting off the identity relation.
# amy7 - W7 = empty set.

define amy7 W7 .o. amy .o. W7;
define bob7 W7 .o. bob .o. W7;

# Bob reflects.
define W8 testdo(W7,bob7,MB,bob%_rfl%_1,bob%_rfl%_O);
# 1 1 0 0 amy_look_m 1 1 0 0 bob_look_m 1 1 0 0 amy_rfl_O 1 1 0 0 bob_rfl_O 1 1 0 0 amy_say_O 1 1 0 0 bob_say_O 1 1 0 0 amy_rfl_1 1 1 1 0 bob_rfl_1 1 1 1 1
# 1 0 0 0 amy_look_f 1 0 0 0 bob_look_m 1 0 0 0 amy_rfl_1 1 0 1 0 bob_rfl_O 1 0 1 0 amy_say_1 1 0 1 0 bob_say_O 1 0 1 0 amy_rfl_1 1 0 1 0 bob_rfl_1 1 0 1 1
# 0 1 0 0 amy_look_m 0 1 0 0 bob_look_f 0 1 0 0 amy_rfl_O 0 1 0 0 bob_rfl_1 0 1 0 1 amy_say_O 0 1 0 1 bob_say_1 0 1 0 1 amy_rfl_1 0 1 1 1 bob_rfl_1 0 1 1 1
define amy8 W8 .o. amy .o. W8;
define bob8 W8 .o. bob .o. W8;

# In the worlds of W8, amy and bob have identified their worlds. The relations amy8 and bob8 are identity relations.
# This is shown by [amy8 - W8] and [bob8 - W8] being empty.

