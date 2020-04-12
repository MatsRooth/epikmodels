source test.fst
source muddy2d-add2.fst

# Select the 01 world from W3
# regex Cn(W3, Nst(MB) & MA);
# 0 1 0 0 al0 0 1 0 0 bl1 0 1 0 0 ar1 0 1 0 1
# Select the 10 world.
# regex Cn(W3, MB & Nst(MA));
# 1 0 0 0 al1 1 0 0 0 bl0 1 0 0 0 ar0 1 0 0 0

# Check if they are connected by relation amy3.
# The result should is empty. This corresponds
# to the absence of the top solid edge in G3.
# Cn(W3, Nst(MB) & MA) .o. amy3 .o. Cn(W3, MB & Nst(MA));

# The same is non-empty in step 0.
# regex Cn(W0, Nst(MB) & MA) .o. amy0 .o. Cn(W0, MB & Nst(MA));
# 0:1 1:0 0 0

# A world ending with St1 is connected by Rn to a world ending with St2 in
# the space of worlds Wn. The value is a bit.
define Connected(Wn,Rn,St1,St2) Test([Cn(Wn,St1) .o. Rn .o. Cn(Wn,St2)].u, "1", "0");

# Try all of the six edges for graph G3.
# 1. Connected(W3,amy3,Nst(MB) & MA,MB & Nst(MA)) => 0
# 2. Connected(W3,amy3,MB & Nst(MA),MB & MA) => 1
# 3. Connected(W3,amy3,MB & MA,Nst(MB) & MA) => 0
# 4. Connected(W3,bob3,Nst(MB) & MA,MB & Nst(MA)) => 0 
# 5. Connected(W3,bob3,MB & Nst(MA),MB & MA) => 0 
# 6. Connected(W3,bob3,MB & MA,Nst(MB) & MA) => 1
# The description of the first six bits of G3 is correct!

# Operator that constructs the six-bit vector describing edges
# given worlds and two Kripke relations. It concatenates the bit
# answers to the six questions above.
define Edge(Wn,R1,R2) Connected(Wn,R1,Nst(MB) & MA,MB & Nst(MA))
  Connected(Wn,R1,MB & Nst(MA),MB & MA)
  Connected(Wn,R1,MB & MA,Nst(MB) & MA)
  Connected(Wn,R2,Nst(MB) & MA,MB & Nst(MA))
  Connected(Wn,R2,MB & Nst(MA),MB & MA)
  Connected(Wn,R2,MB & MA,Nst(MB) & MA);

# Test it to compare results to the page of graph diagrams.
# All are correct.
# regex Edge(W0,amy0,bob0); => 1 1 1 1 1 1 Correct.
# regex Edge(W1,amy1,bob1); => 0 1 0 1 1 1
# regex Edge(W2,amy2,bob2); => 0 1 0 0 0 1
# regex Edge(W3,amy3,bob3); => 0 1 0 0 0 1
# regex Edge(W4,amy4,bob4); => 0 1 0 0 0 1
# regex Edge(W5,amy5,bob5); => 0 1 0 0 0 0
# regex Edge(W6,amy6,bob6); => 0 0 0 0 0 0 Correct.
# regex Edge(W7,amy7,bob7); => 0 0 0 0 0 0
# regex Edge(W8,amy8,bob8); => 0 0 0 0 0 0

# If Wn contains a world ending with St1 then 1 else 0.
define Hasworld(Wn,St1) Test(Cn(Wn,St1),"1","0");

# Find the six reflection bits.
define Reflection(Wn) Hasworld(Wn,MA & Nst(MB) & WA)
 Hasworld(Wn,Nst(MA) & MB & WA)
 Hasworld(Wn,MA & MB & WA)
 Hasworld(Wn,MA & Nst(MB) & WB)
 Hasworld(Wn,Nst(MA) & MB & WB)
 Hasworld(Wn,MA & MB & WB);

# These are the values for W0 through W8;
# W0 0 0 0 0 0 0
# W1 0 0 0 0 0 0
# W2 0 0 0 0 0 0
# W3 1 0 0 0 0 0
# W4 1 0 0 0 1 0
# W5 1 0 0 0 1 0 Speaking step, no change.
# W6 1 0 0 0 1 0
# W7 1 1 1 0 1 0
# W8 1 1 1 1 1 1

# All agree with the graph page [muddy2e].

# Combine the edge vector and reflection to get a representation of the graph.
define Graph(Wn,R1,R2) Edge(Wn,R1,R2) Reflection(Wn);


# Graph(W0,amy0,bob0) => 1 1 1 1 1 1 0 0 0 0 0 0
# Graph(W1,amy1,bob1) => 0 1 0 1 1 1 0 0 0 0 0 0 
# Graph(W2,amy2,bob2) => 0 1 0 0 0 1 0 0 0 0 0 0
# Graph(W3,amy3,bob3) => 0 1 0 0 0 1 1 0 0 0 0 0 
# Graph(W4,amy4,bob4) => 0 1 0 0 0 1 1 0 0 0 1 0
# Graph(W5,amy5,bob5) => 0 1 0 0 0 0 1 0 0 0 1 0
# Graph(W6,amy6,bob6) => 0 0 0 0 0 0 1 0 0 0 1 0
# Graph(W7,amy7,bob7) => 0 0 0 0 0 0 1 1 1 0 1 0
# Graph(W8,amy8,bob8) => 0 0 0 0 0 0 1 1 1 1 1 1

# The above need to be checked.
# The underlying values for Wi, amyi, and bobi were found partially in an ad-hoc
# way. The question then is whether the modal space in the graph semantics can
# be constructed as a KAT model.



