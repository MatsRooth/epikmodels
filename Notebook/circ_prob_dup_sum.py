#!/usr/bin/env python
# coding: utf-8

# Environment Requirement: python 3.8; ipykernel, hfst_dev, graphviz

# ## Setup

# In[1]:


import hfst_dev as hfst
import graphviz
import math


# In[2]:


# Weights are negative log probabilities
print(-math.log(0.25))
print(-math.log(0.75))
print(-math.log(0.5))


# In[3]:


print(-math.log(0.375)) # 3/8
print(-math.log(0.125)) # 1/8
print(-math.log(0.625)) # 5/8
print(-math.log(0.875)) # 7/8


# ## States

# In[4]:


H = hfst.regex('H')
T = hfst.regex('T')
  
# H is unweighted. The initial world H will be iH.
defs = {"H":H,"T":T}

St = hfst.regex('H | T',definitions=defs)
defs.update({"St":St})
St.view()


# In[5]:


# State complement
# X is a compiled machine that is assumed to be a set of states.
def Nst(X):
    nst = St.copy()
    nst.minus(X)
    return(nst)


# In[6]:


# Read this without definitions
UnequalStPair = hfst.regex("[H T] | [T H]")
defs.update({"UnequalStPair":UnequalStPair})
UnequalStPair.view()


# ## Identies 
# Id is the multiplicative identity of the algebra.

# In[7]:


# This is unweighted
# Don't use defs here
Id = hfst.regex('H | T')
Id.view()


# ### Weighted identities
# These can be used to add weights to events, using Cn.

# In[8]:


def weighted_id(num,dem):
    weight = -math.log(num / dem)
    machine = hfst.regex(f'H::{weight} | T::{weight}')
    return(machine)


# In[9]:


Id_1_16 = weighted_id(1,16)
Id_3_16 = weighted_id(3,16)
Id_4_16 = weighted_id(4,16)
Id_1_16.view()
# With Cn, these will have the effect of multiplying the weight of each path by num/dem.


# In[10]:


Id_3_16.view()


# In[11]:


Id_4_16.view()


# ## Actions
# Actions are there in unweighted and weighted versions.  Unweighted is marked with u at the end.
# This defines the unweighted decorated events.

# In[12]:


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



# In[13]:


#accurate announce. Probability will be 4/16.
announceHHu = hfst.regex('[H announceHH H]',definitions=defs)
announceTTu = hfst.regex('[T announceTT T]',definitions=defs)

announceHHu.determinize()
announceTTu.determinize()




# In[14]:


# Flips are biased

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

bflipHTu.view()


# In[15]:


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


# ### Assemble the set of unweighted events

# In[16]:


EventPeekAmyU = hfst.regex('[peekamyHHu | peekamyTTu | peekamyHTu | peekamyTHu ]', definitions=defs)
EventPeekAmyU.determinize()


# In[17]:


EventPeekBobU = hfst.regex('[peekbobHHu | peekbobTTu | peekbobHTu | peekbobTHu ]', definitions=defs)
EventPeekBobU.determinize()


# In[18]:


EventFlipU = hfst.regex('[bflipHHu | bflipTTu | bflipHTu | bflipTHu ]', definitions=defs)
EventFlipU.determinize()


# In[19]:


EventAnnounceU = hfst.regex('[announceHHu | announceTTu ]', definitions=defs)
EventAnnounceU.determinize()


# In[20]:


defs.update({"EventPeekAmyU":EventPeekAmyU,"EventPeekBobU":EventPeekBobU,"EventFliUp":EventFlipU,"EventAnnounceU":EventAnnounceU})


# In[21]:


EventU = EventPeekAmyU.copy()
EventU.disjunct(EventPeekBobU)
EventU.disjunct(EventFlipU)
EventU.disjunct(EventAnnounceU)
EventU.minimize()
defs.update({"EventU":EventU})


# In[22]:


EventU.view()


# In[23]:


# Test -- it should be 14
EventNameU = EventU.extract_paths().keys()
EventNameU = [x[1:-1] for x in EventNameU]
print(EventNameU)
print(len(EventNameU))


# ## Cn in KAT algebra
# This is thought to work both for symbolic and probabilistic arguments.

# In[24]:


# Delete the second state in a block of two states.  This is used defining Ekat concatenation.
Squash = hfst.regex('St -> 0 || St _', definitions = defs)
# Strings that do not contain an unequal state pair
Wf0 = hfst.regex('~[$ UnequalStPair]', definitions = defs)
defs.update({"Squash":Squash,"Wf0":Wf0})


# In[25]:


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


# ## Add weights to the events
# ### Weights of veridical peeks
# They are 3/16, or 1.67 in the log space.

# In[26]:


peekamyHH = Cn(peekamyHHu,Id_3_16)
print(-math.log(3/16))
peekamyHH.view()


# In[27]:


peekamyTT = Cn(peekamyTTu,Id_3_16)
peekamyTT.view()


# In[28]:


peekbobHH = Cn(peekbobHHu,Id_3_16)
peekbobHH.view()


# In[29]:


peekbobTT = Cn(peekbobTTu,Id_3_16)
peekbobTT.view()


# In[30]:


defs.update({"peekamyHH":peekamyHH,"peekamyTT":peekamyTT,"peekbobHH":peekbobHH,"peekbobTT":peekbobTT})


# ### Weights of incorrect peeks
# They are 1/16, or 2.77 in the log space.
# 
# Amy peeks at actual H, perceiving it as T

# In[31]:


peekamyHT = Cn(peekamyHTu,Id_1_16)
print(-math.log(1/16))
peekamyHT.view()


# In[32]:


peekamyTH = Cn(peekamyTHu,Id_1_16)
peekamyTH.view()


# In[33]:


peekbobHT = Cn(peekbobHTu,Id_1_16)
peekbobHT.view()


# In[34]:


peekbobTH = Cn(peekbobTHu,Id_1_16)
peekbobTH.view()


# In[35]:


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


# ## Weights of flips

# In[36]:


# Stays same, weight 3/16
bflipHH = Cn(bflipHHu,Id_3_16)
bflipTT = Cn(bflipTTu,Id_3_16)
# Changes, weight 1/16
bflipHT = Cn(bflipHTu,Id_1_16)
bflipTH = Cn(bflipTHu,Id_1_16)

defs.update({"bflipHH":bflipHH,"bflipTT":bflipTT})


# ## Weights of announcements
# They have weight 1/4, or 4/16.

# In[37]:


announceHH = Cn(announceHHu,Id_4_16)
print(-math.log(4/16))
announceHH.view()


# In[38]:


announceTT = Cn(announceTTu,Id_4_16)
defs.update({"announceHH":announceHH,"announceTT":announceTT})
announceTT.view()


# ## Set of decorated events and subsets thereof
# Assemble this directly with HFST operations. We have indication that the defs
# mechanism strips off weights.

# In[39]:


# Bare event names
eventName = [s[1:-1] for s in EventU.extract_paths().keys()]
eventName


# In[40]:


# Write the code to assemble hEvent with HFST methods
for n in eventName: print(f'hEvent.disjunct({n})')


# In[41]:


hEvent = announceHH.copy()
hEvent.disjunct(bflipHH)
hEvent.disjunct(bflipHT)
hEvent.disjunct(peekamyHH)
hEvent.disjunct(peekamyHT)
hEvent.disjunct(peekbobHH)
hEvent.disjunct(peekbobHT)
hEvent.determinize()
defs.update({"hEvent":hEvent})
hEvent.view()


# In[42]:


# Write the code to assemble hEvent with HFST methods
# Use the last seven.
for n in eventName: print(f'tEvent.disjunct({n})')


# In[43]:


tEvent = announceTT.copy()
tEvent.disjunct(bflipTH)
tEvent.disjunct(bflipTT)
tEvent.disjunct(peekamyTH)
tEvent.disjunct(peekamyTT)
tEvent.disjunct(peekbobTH)
tEvent.disjunct(peekbobTT)
tEvent.determinize()
defs.update({"tEvent":tEvent})
tEvent.view()


# ## Set of weighted events

# In[44]:


Event = hEvent.copy()
Event.disjunct(tEvent)
defs.update({"Event":Event})
Event.determinize()
Event.view()


# ## Worlds of a given length

# In[45]:


# Initial worlds, each with probability 1/2
W0 = weighted_id(1,2)
W0.view()


# In[46]:


W1 = Cn(W0,Event)
W1.view()


# In[47]:


# help(W1.extract_paths)
W1.extract_paths(output='raw')


# In[48]:


W2 = Cn(W1,Event)
W3 = Cn(W2,Event)
W4 = Cn(W3,Event)
W5 = Cn(W4,Event)
W6 = Cn(W5,Event)


# ## Sum weights

# In[49]:


W1.extract_paths(output='raw')


# In[50]:


# This is naive, there should be a better alternative using the forward algorithm.
def total_weight(M):
    total = 0.0
    for weight, path in M.extract_paths(output='raw'):
        total += math.exp(-weight)
    return(total)


# In[51]:


print(f'{total_weight(W0)} {len(W0.extract_paths())}')
print(f'{total_weight(W1)} {len(W1.extract_paths())}')
print(f'{total_weight(W2)} {len(W2.extract_paths())}')
print(f'{total_weight(W3)} {len(W3.extract_paths())}')
print(f'{total_weight(W4)} {len(W4.extract_paths())}')
print(f'{total_weight(W5)} {len(W5.extract_paths())}')
print(f'{total_weight(W6)} {len(W6.extract_paths())}')


# ## Event alternatives

# In[52]:


# Unconstrained cross product of the decorated unweighted events
# This is used in defining event alternative relations.
eventCrossEventU = EventU.copy()
eventCrossEventU.cross_product(EventU)

defs.update({"eventCrossEventU": eventCrossEventU})
# Shorter name
defs.update({"eCe": eventCrossEventU})


# In[53]:


def crossRel(upper,lower):
    U = hfst.regex('a & b')
    for m in upper: U.disjunct(m)
    L = hfst.regex('a & b')
    for m in lower: L.disjunct(m)
    U.cross_product(L)
    U.determinize()
    return(U)


# In[54]:


X = crossRel([peekamyHHu,peekamyTHu],[peekamyHHu,peekamyTHu])
X.view()


# In[55]:


# Amy's alternatives for Amy's own peeks
Xh = crossRel([peekamyHHu,peekamyTHu],[peekamyHHu,peekamyTHu])
Xt = crossRel([peekamyHTu,peekamyTTu],[peekamyHTu,peekamyTTu])
Xh.view()


# In[56]:


Xt.view()


# In[57]:


amyAmyPeekEventRel = Xh.copy()
amyAmyPeekEventRel.disjunct(Xt)
amyAmyPeekEventRel.determinize()
amyAmyPeekEventRel.view()


# In[58]:


# Amy's alternatives for Bob's peeks
# is a product of four events on each side, she just knows Bob is peeking.
X = [peekbobHHu,peekbobHTu,peekbobTHu,peekbobTTu]
amyBobPeekEventRel = crossRel(X,X)
amyBobPeekEventRel.determinize()
amyBobPeekEventRel.view()


# In[59]:


# Bob's alternatives for Bob's peeks
# Bob sensing H
X = [peekbobHHu,peekbobTHu]
bobBobPeekEventRel = crossRel(X,X)
# Bob sensing T
X = [peekbobHTu,peekbobTTu]
bobBobPeekEventRel.disjunct(crossRel(X,X))
bobBobPeekEventRel.determinize()
bobBobPeekEventRel.view()


# In[60]:


# Bob's alternatives for Amy's peeks
# is a product of four events on each side, she just knows Bob is peeking.
X = [peekamyHHu,peekamyHTu,peekamyTHu,peekamyTTu]
bobAmyPeekEventRel = crossRel(X,X)
bobAmyPeekEventRel.determinize()
bobAmyPeekEventRel.view()


# In[61]:


# Assemble alternative relation on peeks
amyPeekRel = amyAmyPeekEventRel.copy()
amyPeekRel.disjunct(amyBobPeekEventRel)

bobPeekRel = bobBobPeekEventRel.copy()
bobPeekRel.disjunct(bobAmyPeekEventRel)


# ### Alternative relations for flips
# Agents have no constraints

# In[62]:


X = [bflipHHu,bflipHTu,bflipTHu,bflipTTu]
amyFlipRel = crossRel(X,X)
bobFlipRel = crossRel(X,X)
amyFlipRel.view()


# ### Alternative relations for announcements

# In[63]:


amyAnnounceRel = announceHHu.copy()
amyAnnounceRel.disjunct(announceTTu)
amyAnnounceRel.determinize()
amyAnnounceRel.view()


# In[64]:


bobAnnounceRel = amyAnnounceRel.copy()


# ### Assemble decorated event alternative relations

# In[65]:


amyEventRel = amyPeekRel.copy()
amyEventRel.disjunct(amyFlipRel)
amyEventRel.disjunct(amyAnnounceRel)
amyEventRel.determinize()


# In[66]:


bobEventRel = bobPeekRel.copy()
bobEventRel.disjunct(bobFlipRel)
bobEventRel.disjunct(bobAnnounceRel)
bobEventRel.determinize()


# In[67]:


amyEventRel.view()


# ## Test the event relations
# For each agent and the 14 events, display the decorated event alternatives
# as an acceptor.

# In[68]:


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


# In[69]:


k = 1
for ename in EventNameU:
    ename = ename + "u"
    e = eval(ename)
    image = amyEventImage(e)
    print(f'{k}. Amy\'s alternatives to {ename}')
    display(image.view())
    k = k + 1


# In[70]:


k = 1
for ename in EventNameU:
    ename = ename + "u"
    e = eval(ename)
    image = bobEventImage(e)
    print(f'{k}. Bob\'s alternatives to {ename}')
    display(image.view())
    k = k + 1


# The above looks good.
# ## Epik operations
# These are defined using HFST methods.
# Product Cn was defined above.

# In[71]:


# define Kpl(X) [[[X+] & Wf0] .o. Squash].l;
def Kpl(X):
    Z = X.copy()
    Z.repeat_plus()
    Z.intersect(Wf0)
    Z.compose(Squash)
    Z.output_project()
    Z.determinize()
    # Z.minimize()
    return Z


# In[72]:


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


# In[73]:


# Inverse of Squash
Squashi = Squash.copy()
Squashi.invert()


# Wf0 was defined in the section defining Cn.

# In[74]:


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


# In[75]:


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


# In[76]:


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


# ## Unweighted world alterative relations
# Unweighted Amy world alternative relation.

# In[77]:


# Amy world alternative relation. 
amyU = RelKst(amyEventRel)
def amyImage(w):
    result = w.copy()
    result.compose(amyU)
    result.output_project()
    return(result)


# In[78]:


amyImage(peekamyTTu).view()


# In[79]:


k = 1
for dname in EventNameU:
    dname = dname + "u"
    d = eval(dname)
    for ename in EventNameU:
        ename = ename + "u"
        e = eval(ename)
        de = Cn(d,e)
        print(f'{k}. Amy\'s alternatives to {dname} {ename}')
        image = amyImage(de)
        display(image.view())
        k = k + 1


# ## Weighted worlds of unconstrained length
# We already have W0 defined as follows.
# 
# Initial worlds, each with probability 1/2
# W0 = weighted_id(1,2)

# In[80]:


EventPlus = Event.copy()
EventPlus.repeat_plus() 
World = Cn(W0,EventPlus)


# In[81]:


two = hfst.regex("?^{5,5}")
two.intersect(World)
two.view()


# In[82]:


total_weight(two)


# In[83]:


# Find all worlds and their sum probability by
# repeated concatenation.
WN = W0.copy()
for i in range(0,8):
    print(f'{i} {total_weight(WN)} {len(WN.extract_paths())}')
    WN = Cn(WN,Event)

               


# In[84]:


# Find all worlds and their sum probability by
# intersection with World. 
for i in range(1,8):
    n = 2 * i + 1
    M = hfst.regex(f'?^{{{n},{n}}}')
    M.intersect(World)
    print(f'{i} {total_weight(M)} {len(M.extract_paths())}')


# ## Now try the big idea

# In[85]:


amy = amyU.copy()
amy.compose(World)
def amyWImage(w):
    result = w.copy()
    result.compose(amy)
    result.output_project()
    return(result)


# In[86]:


k = 1
for dname in EventNameU:
    dname = dname + "u"
    d = eval(dname)
    for ename in EventNameU:
        ename = ename + "u"
        e = eval(ename)
        de = Cn(d,e)
        print(f'{k}. Amy\'s unnormalized weighted alternatives to {dname} {ename}')
        image = amyWImage(de)
        image.push_weights_to_end()
        display(image.view())
        # See if we can sum to the input
        print('  Attempted sum:')
        de.compose(amyU)
        de.compose(image)
        de.input_project()
        de.push_weights_to_end()
        display(de.view())
        de.determinize()
        print('  Determinized attempted sum:')
        display(de.view())
        k = k + 1


# In[87]:


hfst.HfstTransducer.is_implementation_type_available(
    hfst.ImplementationType.LOG_OPENFST_TYPE)


# ```
# X_log = hfst.HfstTransducer(X)     # copy
# X_log.convert(hfst.ImplementationType.LOG_OPENFST_TYPE)
# ```
# Look at the example announceHHu bflipHHu

# In[88]:


w = Cn(announceHHu,bflipHHu)
w.minimize()
print('Base world')
display(WA.view())
image = amyWImage(w)
image.push_weights_to_end()
print('Amy weighted alternatives. bflipHH is more probable.')
display(image.view())
print('Weighted alternatives mapped back to base.')
ww = w.copy()
ww.compose(amyU)
ww.compose(image)
ww.input_project()
ww.push_weights_to_end()
display(ww.view())
print(' ... and summed.')
ww.convert(hfst.ImplementationType.LOG_OPENFST_TYPE)
ww.determinize()
display(ww.view())


# In[ ]:




