module Syntax where
-- To test it
--  ghc -o SyntaxA SyntaxA.hs
--  ./SyntaxA > m1.fst

import Data.Map (Map)
import qualified Data.Map as Map

import Data.Set (Set)
import qualified Data.Set as Set
 
data EventPrimitive = EventPrimitive String deriving (Eq, Ord)
data TestPrimitive = TestPrimitive String deriving (Eq, Ord)

instance Show EventPrimitive where
  show (EventPrimitive s) = s

instance Show TestPrimitive where
  show (TestPrimitive s) = s

-- KAT formula
data Kat =
  KZero
  | KOne 
  | KTest Test -- KTest :: Test -> Kat
  | KEvent EventPrimitive -- KVar :: EventPrimitive -> Kat
  | KProduct Kat Kat       -- KSeq :: Kat -> Kat -> Kat  
  | KPlus Kat Kat     -- KUnion :: Kat -> Kat -> Kat
  | KAnd Kat Kat 
  | KStar Kat          -- KStar :: Kat -> Kat
  deriving (Eq, Ord)

-- This is not used in generating Fst.
instance Show Kat where
  show KZero = "0"
  show KOne = "1"
  show (KTest t) = "(" ++ show t ++ ")"
  show (KEvent s) = show s
  show (KProduct p q) = show p ++ " " ++ show q
  show (KPlus p q) = "(" ++ show p ++ " + " ++ show q ++ ")"
  show (KAnd p q) = "(" ++ show p ++ " & " ++ show q ++ ")"
  show (KStar p) = "(" ++ show p ++ ")*"

-- State formula as in Figure 1.
data Test =
  TAtom TestPrimitive
  | TFalse
  | TTrue
  | TOr Test Test
  | TAnd Test Test
  | TNeg Test
  deriving (Eq,Ord)

-- Notation follows Fig. 1, e.g TAnd
-- is shown with juxtaposition.
-- Except negation is overbar in Fig. 1.
-- This is not used in generating Fst.
instance Show Test where
  show (TAtom v) = show v
  show TFalse = "0"
  show TTrue = "1"
  show (TOr p q) = show p ++ " + " ++ show q
  show (TAnd p q) = show p ++ show q
  show (TNeg x) = "~" ++ show x

-- Effect formulas as in Figure 1.
data Effect =
  EPair Test Test
  | EOr Effect Effect
  | EAnd Effect Effect
  | ENeg Effect
  deriving (Eq,Ord)

-- This is not used in generating Fst.
instance Show Effect where
  show (EPair u v) = show u ++ " : " ++ show v
  show (EOr u v) = show u ++ " + " ++ show v
  show (EAnd u v) = show u ++ " & " ++ show v
  show (ENeg x) = "~" ++ show x

type Agent = String

data ModelSpecification =
   ModelSpecification { alphabet :: Set TestPrimitive        -- test alphabet
           , assertions :: [Test] -- conditions that specify consistent atoms
           , events :: [(EventPrimitive, Effect)] -- events and their effect formulas
           , agents :: [(Agent, Map EventPrimitive [EventPrimitive])] -- agents and their bare event
                                                                    -- alternative relations.
           } deriving (Eq,Show)

-- combineDecls decl decl' =
--  let join j f = f decl `j` f decl' in
--  Program { alphabet= join (Set.union) alphabet
--          , assertions = join (++) assertions
--          , events = join (++) events
--         , agents = join  (++) agents

-- Construct the alphabet
-- tests ["h","t","h"] ~> fromList [h,t] :: Set TestPrimitive
tests :: [String] -> Set TestPrimitive
tests [] = Set.empty
tests (a : as) = Set.insert (TestPrimitive a) (tests as)


-- Construct effect formulas for tests remaining constant etc.

constantEffect :: TestPrimitive -> Effect
constantEffect a = EOr (EPair (TAtom a) (TAtom a)) (EPair (TNeg (TAtom a)) (TNeg (TAtom a)))
-- constantEffect (TestPrimitive "h")  -> h : h + ~h : ~h

-- Precondition that fluent a is true
-- preTrue (TestPrimitive "h") ~> h : 1
preTrue :: TestPrimitive -> Effect
preTrue a = EPair (TAtom a) TTrue


-- Precondition that fluent a is false
preFalse :: TestPrimitive -> Effect
preFalse a = EPair (TNeg (TAtom a)) TTrue

-- All the fluents in the set are constant.
-- constantEffects (tests ["c1","c2","c3"])  ~> c1 : c1 + ~c1 : ~c1 & c2 : c2 + ~c2 : ~c2 & c3 : c3 + ~c3 : ~c3
constantEffects :: (Set TestPrimitive) -> Effect
constantEffects as = let bs = (Set.toList as)
  in (constantEffects0 bs)
   where constantEffects0 [x] = constantEffect x
         constantEffects0 (x : xs) = EAnd (constantEffect x) (constantEffects0 xs)
         constantEffects0 [] = EPair TTrue TTrue

-- Construct a list of Fst declarations that define
--  Bool
--  St0 primitive atoms
--  each fluent in set the set as
--  St atoms restricted by state formula phi
testDef :: (Set TestPrimitive) -> Test -> [String]
testDef as phi = ["define Bool [\"0\" | \"1\"];",                    -- Bool
                     "define St0 Bool^" ++ (show (length as)) ++ ";",   -- St0
                     "define St St0;"]                                  -- temporary St
                 ++ (testDef2 (Set.toList as))                       -- fluents
                 ++ ["define St St0 & " ++ (testFst phi) ++ ";"]        -- St
                 ++ [(unequalStPair as)]
  where testDef0 :: [TestPrimitive] -> TestPrimitive -> String
        testDef0 [y] x = (if (x == y) then "1" else "Bool")
        testDef0 (y : xs) x = (if (x == y) then "1" else "Bool") ++ " " ++ (testDef0 xs x)
        testDef1 :: [TestPrimitive] -> TestPrimitive -> String
        testDef1 ys x = "define " ++ (show x) ++ " [" ++ (testDef0 ys x) ++ "];"
        testDef2 :: [TestPrimitive] -> [String]
        testDef2 ys = map (testDef1 ys) ys

-- Example
-- testDef (tests ["c1","c2","c3"])

-- Map a state formula to Fst syntax in a context where St and the fluents are defined.
testFst :: Test -> String 
testFst (TAtom a) = (show a)
testFst TFalse = "[St - St]"
testFst TTrue = "St"
testFst (TOr x y) = "[" ++ (testFst x) ++ " | " ++ (testFst y) ++ "]"
testFst (TAnd x y) = "[" ++ (testFst x) ++ " & " ++ (testFst y) ++ "]"
testFst (TNeg x) = "[St - " ++ (testFst x) ++ "]"

-- For examples
-- testFst phi0 ~> "[[h | t] & [St - [h & t]]]"
-- testDef b0 phi0 ~>
phi0 = (TAnd (TOr (TAtom (TestPrimitive "h")) (TAtom (TestPrimitive "t"))) (TNeg (TAnd (TAtom (TestPrimitive "h")) (TAtom (TestPrimitive "t")))))
b0 = (tests ["h","t"])



-- Fst string with the form of a pair of non-matching atoms.
-- unequalStPair b0 ~> "define UnequalStPair [h [St - h]] | [[St - h] h] | [t [St - t]] | [[St - t] t];"
unequalStPair :: (Set TestPrimitive) -> String
unequalStPair xs = "define UnequalStPair " ++ (disjoin (map unequalStPair0 (Set.toList b0))) ++ ";"
  where disjoin [x] = x
        disjoin (x : xs) = x ++ " | " ++ (disjoin xs)

unequalStPair0 :: TestPrimitive -> String
unequalStPair0 x = "[" ++ (testFst x') ++ " " ++ (testFst (TNeg x')) ++ "] | [" ++
             (testFst (TNeg x')) ++ " " ++ (testFst x') ++ "]"
  where x' = (TAtom x)
  



-- Definitions of event types from their effect formulas.
-- Define Event as the disjunction of the event types.

effectDef :: [(EventPrimitive, Effect)] -> [String]
effectDef xs = (map effectDef1 xs) ++ ["define Event " ++ (disjoin (map fst xs)) ++ ";"]
  where disjoin [x] = (show x)
        disjoin (x : xs) = (show x) ++ " | " ++ (disjoin xs)
 

effectDef1 :: (EventPrimitive, Effect) -> String
effectDef1 (e, eta) = "define " ++ (show e) ++ " " ++ (effectDef0 eta) ++ " & [St " ++ (show e) ++ " St];"
  where effectDef0 :: Effect -> String
        effectDef0 (EPair u v) = "[" ++ (testFst u) ++ " ? " ++ (testFst v) ++ "]"
        effectDef0 (EOr x y) = "[" ++ (effectDef0 x) ++ " | " ++ (effectDef0 y) ++ "]"
        effectDef0 (EAnd x y) = "[" ++ (effectDef0 x) ++ " & " ++ (effectDef0 y) ++ "]"
        effectDef0 (ENeg x) = "[[St ? St] - " ++ (effectDef0 x) ++ "]"

eta_a1 =  EAnd (preTrue (TestPrimitive "h")) (constantEffects (tests ["h","t"]))

evspec = [((EventPrimitive "a1"), EAnd (preTrue (TestPrimitive "h")) (constantEffects (tests ["h","t"]))),
          ((EventPrimitive "a0"), EAnd (preFalse (TestPrimitive "h")) (constantEffects (tests ["h","t"]))),
          ((EventPrimitive "b1"), EAnd (preTrue (TestPrimitive "h")) (constantEffects (tests ["h","t"]))),
          ((EventPrimitive "b0"), EAnd (preFalse (TestPrimitive "h")) (constantEffects (tests ["h","t"])))]

-- Definitions of agent epistemic alterntive relations.
-- Kat.fst defines KAT operations such as RelKst in Fst.



-- (Agent, [(EventPrimitive, [EventPrimitive])])


eventRel0 :: (EventPrimitive,[EventPrimitive]) -> String
eventRel0 (e,es) = "[" ++ (show e) ++ " .x. [" ++ (disjoin es) ++ "]]"
  where disjoin [x] = (show x)
        disjoin (x : xs) = (show x) ++ " | " ++ (disjoin xs)

eventRel1 :: [(EventPrimitive,[EventPrimitive])] -> String
eventRel1 xs = "[" ++ (disjoin (map eventRel0 xs)) ++ "]"
  where disjoin [x] = x
        disjoin (x : xs) = x ++ " | " ++ (disjoin xs)

-- effect ((EventPrimitive "a1"), eta_a1)

amyspec = [(EventPrimitive "a1", map EventPrimitive ["a1"]),
           (EventPrimitive "a0", map EventPrimitive ["a0"]),
           (EventPrimitive "b1", map EventPrimitive ["b1","b0"]),
           (EventPrimitive "b0", map EventPrimitive ["b1","b0"])]

bobspec = [(EventPrimitive "b1", map EventPrimitive ["b1"]),
           (EventPrimitive "b0", map EventPrimitive ["b0"]),
           (EventPrimitive "a1", map EventPrimitive ["a1","a0"]),
           (EventPrimitive "a0", map EventPrimitive ["a1","a0"])]          

agentspec = [("amy",amyspec),("bob",bobspec)]

agentDef :: [(Agent, [(EventPrimitive, [EventPrimitive])])] -> [String]
agentDef xs = ["source kat.fst;"] ++ (map agentDef0 xs)

agentDef0 :: (Agent, [(EventPrimitive, [EventPrimitive])]) -> String
agentDef0 (a,xs) = "define " ++ a ++ " RelKst(" ++ (eventRel1 xs) ++ ");"

-- main = do let out1 = unlines (testDef b0 phi0)
--          putStrLn out1
--          let out2  = effect ((EventPrimitive "a1"), eta_a1)
--          putStrLn out2

