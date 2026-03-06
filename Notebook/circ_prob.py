#!/usr/bin/env python
# coding: utf-8


import hfst_dev as hfst
import graphviz
import math

H = hfst.regex('H')
T = hfst.regex('T')
  
# H is unweighted. The initial world H will be iH.
defs = {"H":H,"T":T}

St = hfst.regex('H | T',definitions=defs)
defs.update({"St":St})
St.view()

def Nst(X):
    nst = St.copy()
    nst.minus(X)
    return(nst)

# Read this without definitions
UnequalStPair = hfst.regex("[H T] | [T H]")
defs.update({"UnequalStPair":UnequalStPair})



# ## Identies 
# Id is the multiplicative identity of the algebra.
# This is unweighted
# Don't use defs here
# This seems to duplicate St, do we need St?
Id = hfst.regex('H | T')


# ### Weighted identities
# With Cn, these will have the effect of multiplying the weight of each path by num/dem.
# Or the log equivalent.

def weighted_id(num,dem):
    weight = -math.log(num / dem)
    machine = hfst.regex(f'H::{weight} | T::{weight}')
    return(machine)

Id_1_16 = weighted_id(1,16)
Id_3_16 = weighted_id(3,16)
Id_4_16 = weighted_id(4,16)


# ## Actions
# Actions are there in unweighted and weighted versions.  Unweighted is marked with u at the end.
# This defines the unweighted decorated events.


# Peeking is going to be construed as fallible
# Accurate peeks, probability will be 3/16
peekamyHHu = hfst.regex('[H peekamyHH H]',definitions=defs) # Amy peeks at H, perceiving H
peekamyTTu = hfst.regex('[T peekamyTT T]',definitions=defs)

# Inaccurate peeks, probability will be 1/16
peekamyHTu = hfst.regex('[H peekamyHT H]',definitions=defs) # Amy peeks at H, perceiving T
peekamyTHu = hfst.regex('[T peekamyTH T]',definitions=defs)

# Accurate
peekbobHHu = hfst.regex('[H peekbobHH H]',definitions=defs) # Bob peeks at H, perceiving H
peekbobTTu = hfst.regex('[T peekbobTT T]',definitions=defs)

# Inaccurate
peekbobHTu = hfst.regex('[H peekbobHT H]',definitions=defs)  # Bob peeks at H, perceiving T
peekbobTHu = hfst.regex('[T peekbobTH T]',definitions=defs)

peekamyHHu.determinize()
peekamyTTu.determinize()
peekamyHTu.determinize()
peekamyTHu.determinize()

peekbobHHu.determinize()
peekbobTTu.determinize()
peekbobHTu.determinize()
peekbobTHu.determinize()

#accurate announce. Probability will be 4/16.
announceHHu = hfst.regex('[H announceHH H]',definitions=defs)
announceTTu = hfst.regex('[T announceTT T]',definitions=defs)

announceHHu.determinize()
announceTTu.determinize()

# Flips will be biased

# Coin stays the same. The weighted version will have a higher probability of 3/16
bflipHHu =  hfst.regex('[H bflipHH H ]',definitions=defs)
bflipTTu =  hfst.regex('[T bflipTT T ]',definitions=defs)

# Coin changes, lower probability of 1/16
bflipHTu =  hfst.regex('[H bflipHT T ]',definitions=defs)
bflipTHu =  hfst.regex('[T bflipTH H ]',definitions=defs)

bflipHHu.determinize()
bflipTTu.determinize()   
bflipHTu.determinize()
bflipTHu.determinize()

# These name the 14 unweighted decorated events
eventdefs_u = { 
    "peekamyHHu":peekamyHHu,
    "peekamyTTu":peekamyTTu,
    "peekamyHTu":peekamyHTu,
    "peekamyTHu":peekamyTHu,
    "peekbobHHu":peekbobHHu,
    "peekbobTTu":peekbobTTu,
    "peekbobHTu":peekbobHTu,
    "peekbobTHu":peekbobTHu,
    "announceHHu":announceHHu,
    "announceTTu":announceTTu,
    "bflipHHu":bflipHHu,
    "bflipTTu":bflipTTu, 
    "bflipHTu":bflipHTu,
    "bflipTHu":bflipTHu }
defs.update(eventdefs_u)

## Assemble sets of unweighted events

EventPeekAmyU = hfst.regex('[peekamyHHu | peekamyTTu | peekamyHTu | peekamyTHu ]', definitions=defs)
EventPeekAmyU.determinize()

EventPeekBobU = hfst.regex('[peekbobHHu | peekbobTTu | peekbobHTu | peekbobTHu ]', definitions=defs)
EventPeekBobU.determinize()

EventFlipU = hfst.regex('[bflipHHu | bflipTTu | bflipHTu | bflipTHu ]', definitions=defs)
EventFlipU.determinize()

EventAnnounceU = hfst.regex('[announceHHu | announceTTu ]', definitions=defs)
EventAnnounceU.determinize()

defs.update({"EventPeekAmyU":EventPeekAmyU,"EventPeekBobU":EventPeekBobU,"EventFliUp":EventFlipU,"EventAnnounceU":EventAnnounceU})

EventU = EventPeekAmyU.copy()
EventU.disjunct(EventPeekBobU)
EventU.disjunct(EventFlipU)
EventU.disjunct(EventAnnounceU)
EventU.minimize()
defs.update({"EventU":EventU})


EventNameU = EventU.extract_paths().keys()
EventNameU = [x[1:-1] for x in EventNameU]

# Test -- it should be 14
# print(EventNameU)
# print(len(EventNameU))


# ## Cn in KAT algebra
# This is thought to work both for symbolic and probabilistic arguments.

# Delete the second state in a block of two states.  This is used defining Ekat concatenation.
Squash = hfst.regex('St -> 0 || St _', definitions = defs)

# Strings that do not contain an unequal state pair
Wf0 = hfst.regex('~[$ UnequalStPair]', definitions = defs)
defs.update({"Squash":Squash,"Wf0":Wf0})


# define Cn(X,Y) [[[X Y] & Wf0] .o. Squash].l;
def Cn(X,Y):
    Z = X.copy()
    Z.concatenate(Y)
    Z.intersect(Wf0)
    Z.compose(Squash)
    Z.output_project()
    #Z.determinize() or Z.minimize() or both? 
    Z.minimize()
    return Z


## Add weights to the events
### Weights of veridical peeks
# They are 3/16, or 1.67 in the log space.

peekamyHH = Cn(peekamyHHu,Id_3_16)
peekamyTT = Cn(peekamyTTu,Id_3_16)

peekbobHH = Cn(peekbobHHu,Id_3_16)
peekbobTT = Cn(peekbobTTu,Id_3_16)

defs.update({"peekamyHH":peekamyHH,"peekamyTT":peekamyTT,"peekbobHH":peekbobHH,"peekbobTT":peekbobTT})

## Incorrect peeks
# They are 1/16, or 2.77 in the log space.
# Amy peeks at actual H, perceiving it as T

peekamyHT = Cn(peekamyHTu,Id_1_16)
peekamyTH = Cn(peekamyTHu,Id_1_16)

peekbobHT = Cn(peekbobHTu,Id_1_16)
peekbobTH = Cn(peekbobTHu,Id_1_16)

defs.update({ 
    "peekamyHH":peekamyHH,
    "peekamyTT":peekamyTT,
    "peekamyHT":peekamyHT,
    "peekamyTH":peekamyTH })
defs.update({ 
    "peekbobHH":peekbobHH,
    "peekbobTT":peekbobTT,
    "peekbobHT":peekbobHT,
    "peekbobTH":peekbobTH })


## Flips
# Stays same, weight 3/16
bflipHH = Cn(bflipHHu,Id_3_16)
bflipTT = Cn(bflipTTu,Id_3_16)

# Changes, weight 1/16
bflipHT = Cn(bflipHTu,Id_1_16)
bflipTH = Cn(bflipTHu,Id_1_16)

defs.update({"bflipHH":bflipHH,"bflipTT":bflipTT,"bflipHT":bflipHT,"bflipTH":bflipTH})


# Snnouncements
# They have weight 1/4, or 4/16.

announceHH = Cn(announceHHu,Id_4_16)
announceTT = Cn(announceTTu,Id_4_16)

defs.update({"announceHH":announceHH,"announceTT":announceTT})


### Set of decorated events and subsets thereof
# Assemble this directly with HFST operations. We have indication that the defs
# mechanism strips off weights.

# Bare event names
eventName = [s[1:-1] for s in EventU.extract_paths().keys()]

# Write the code to assemble hEvent with HFST methods
# for n in eventName: print(f'hEvent.disjunct({n})')

hEvent = announceHH.copy()
hEvent.disjunct(bflipHH)
hEvent.disjunct(bflipHT)
hEvent.disjunct(peekamyHH)
hEvent.disjunct(peekamyHT)
hEvent.disjunct(peekbobHH)
hEvent.disjunct(peekbobHT)
hEvent.determinize()

defs.update({"hEvent":hEvent})


# Write the code to assemble hEvent with HFST methods
# Use the last seven.
# for n in eventName: print(f'tEvent.disjunct({n})')

tEvent = announceTT.copy()
tEvent.disjunct(bflipTH)
tEvent.disjunct(bflipTT)
tEvent.disjunct(peekamyTH)
tEvent.disjunct(peekamyTT)
tEvent.disjunct(peekbobTH)
tEvent.disjunct(peekbobTT)
tEvent.determinize()

defs.update({"tEvent":tEvent})

Event = hEvent.copy()
Event.disjunct(tEvent)

defs.update({"Event":Event})
Event.determinize()


### Worlds of a given length
W0 = weighted_id(1,2)
W1 = Cn(W0,Event)
W2 = Cn(W1,Event)
W3 = Cn(W2,Event)
W4 = Cn(W3,Event)
W5 = Cn(W4,Event)
W6 = Cn(W5,Event)


### Sum weights

# This is naive, there should be a better alternative using the forward algorithm.
def total_weight(M):
    total = 0.0
    for weight, path in M.extract_paths(output='raw'):
        total += math.exp(-weight)
    return(total)

# Unconstrained cross product of the decorated unweighted events
# This is used in defining event alternative relations.
eventCrossEventU = EventU.copy()
eventCrossEventU.cross_product(EventU)

defs.update({"eventCrossEventU": eventCrossEventU})
# Shorter name
defs.update({"eCe": eventCrossEventU})

## The inputs are lists of machines
def crossRel(upper,lower):
    U = hfst.regex('a & b')
    for m in upper: U.disjunct(m)
    L = hfst.regex('a & b')
    for m in lower: L.disjunct(m)
    U.cross_product(L)
    U.determinize()
    return(U)

# X = crossRel([peekamyHHu,peekamyTHu],[peekamyHHu,peekamyTHu])
# X.view()

# Amy's alternatives for Amy's own peeks
Xh = crossRel([peekamyHHu,peekamyTHu],[peekamyHHu,peekamyTHu])
Xt = crossRel([peekamyHTu,peekamyTTu],[peekamyHTu,peekamyTTu])

amyAmyPeekEventRel = Xh.copy()
amyAmyPeekEventRel.disjunct(Xt)
amyAmyPeekEventRel.determinize()

# Amy's alternatives for Bob's peeks
# is a product of four events on each side, she just knows Bob is peeking.
X = [peekbobHHu,peekbobHTu,peekbobTHu,peekbobTTu]
amyBobPeekEventRel = crossRel(X,X)
amyBobPeekEventRel.determinize()



# Bob's alternatives for Bob's peeks
# Bob sensing H
X = [peekbobHHu,peekbobTHu]
bobBobPeekEventRel = crossRel(X,X)

# Bob sensing T
X = [peekbobHTu,peekbobTTu]
bobBobPeekEventRel.disjunct(crossRel(X,X))
bobBobPeekEventRel.determinize()

# Bob's alternatives for Amy's peeks
# is a product of four events on each side, she just knows Bob is peeking.
X = [peekamyHHu,peekamyHTu,peekamyTHu,peekamyTTu]
bobAmyPeekEventRel = crossRel(X,X)
bobAmyPeekEventRel.determinize()

# Assemble alternative relation on peeks
amyPeekRel = amyAmyPeekEventRel.copy()
amyPeekRel.disjunct(amyBobPeekEventRel)

bobPeekRel = bobBobPeekEventRel.copy()
bobPeekRel.disjunct(bobAmyPeekEventRel)


# ### Alternative relations for flips
# Agents have no constraints

X = [bflipHHu,bflipHTu,bflipTHu,bflipTTu]
amyFlipRel = crossRel(X,X)
bobFlipRel = crossRel(X,X)

### Alternative relations for announcements

amyAnnounceRel = announceHHu.copy()
amyAnnounceRel.disjunct(announceTTu)
amyAnnounceRel.determinize()

bobAnnounceRel = amyAnnounceRel.copy()

amyEventRel = amyPeekRel.copy()
amyEventRel.disjunct(amyFlipRel)
amyEventRel.disjunct(amyAnnounceRel)
amyEventRel.determinize()

bobEventRel = bobPeekRel.copy()
bobEventRel.disjunct(bobFlipRel)
bobEventRel.disjunct(bobAnnounceRel)
bobEventRel.determinize()

# For each agent and the 14 events, display the decorated event alternatives
# as an acceptor.

def amyEventImage(e):
    result = e.copy()
    result.compose(amyEventRel)
    result.output_project()
    return(result)
# NB the argument should be unweighted for this test

def bobEventImage(e):
    result = e.copy()
    result.compose(bobEventRel)
    result.output_project()
    return(result)

## Algebraic operations of Epikat. Cn was defined earlier.

def Kpl(X):
    Z = X.copy()
    Z.repeat_plus()
    Z.intersect(Wf0)
    Z.compose(Squash)
    Z.output_project()
    Z.determinize()
    # Z.minimize()
    return Z


# Kleene Star
# The identity is St, not the empty string.
# define Kst(X) St |  Kpl(X);
# Here the identity is Id.
def Kst(X):
    Z = Kpl(X)
    Z.disjunct(Id)
    Z.determinize()
    # Z.minimize()
    return(Z)

# Inverse of Squash
Squashi = Squash.copy()
Squashi.invert()

# Concatenation product of relations
# NB this in not relation composition.
# define Cnr(R,S) Squash.i .o. Wf0 .o. [R S] .o. Wf0 .o. Squash;
def Cnr(R,S):
    Z = R.copy()
    Z.concatenate(S)
    Z.compose(Wf0)
    Z.compose(Squash)
    Z.invert()
    Z.compose(Wf0)
    Z.compose(Squash)
    Z.invert()
    Z.determinize()
    return(Z)


# Kleene plus on relations
# define RelKpl(X) Squash.i .o. Wf0 .o. [X+] .o. Wf0 .o. Squash;
def RelKpl(R):
    Z = R.copy()
    Z.repeat_plus() 
    Z.compose(Wf0)
    Z.compose(Squash)
    Z.invert()
    Z.compose(Wf0)
    Z.compose(Squash)
    Z.invert()
    Z.determinize()
    return(Z)

# Kleene star on relations
# This is used in defining world alternative relations.
# The total relation on St is included.
# define RelKst(X) [St .x. St] | RelKpl(X);
def RelKst(R):
    Z = RelKpl(R)
    IdxId = Id.copy()
    IdxId.cross_product(St)
    Z.disjunct(IdxId)
    return(Z)

# Amy unweighted world alternative relation. 
amyU = RelKst(amyEventRel)
def amyImage(w):
    result = w.copy()
    result.compose(amyU)
    result.output_project()
    return(result)

# ## Weighted worlds of unconstrained length
# We already have W0 defined as follows.
# 
# Initial worlds, each with probability 1/2
# W0 = weighted_id(1,2)

EventPlus = Event.copy()
EventPlus.repeat_plus() 
World = Cn(W0,EventPlus)

# Find all worlds and their sum probability by
# repeated concatenation.
WN = W0.copy()
for i in range(0,8):
    print(f'{i} {total_weight(WN)} {len(WN.extract_paths())}')
    WN = Cn(WN,Event)

# The big idea
# Weighed Kripke relations are defined by composing
# unweighted ones with the probabilistic circumstantial world model.

amy = amyU.copy()
amy.compose(World)

def amyWImage(w):
    result = w.copy()
    result.compose(amy)
    result.output_project()
    return(result)

bob = amyU.copy()
bob.compose(World)

def bobWImage(w):
    result = w.copy()
    result.compose(amy)
    result.output_project()
    return(result)

#hfst.HfstTransducer.is_implementation_type_available(
#    hfst.ImplementationType.LOG_OPENFST_TYPE)


# ```
# X_log = hfst.HfstTransducer(X)     # copy
# X_log.convert(hfst.ImplementationType.LOG_OPENFST_TYPE)
# ```
# Look at the example announceHHu bflipHHu






