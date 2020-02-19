{
import Token
-- As in Epik
-- import Data.Set (Set)
-- import qualified Data.Set as Set
import System.IO
import System.Environment
import Prelude
import Data.List.Utils (replace)
}

-- Entry point. This function is used in Calc.hs as (parseCalc s).
%name parseEpik

-- This corresponds to the type Token defined in Token.x.
%tokentype { Token }
%error { parseError }

-- The guys in the first column are used in grammar rules for Exp seen below.
-- The guys in the curly brackets correspond to the kinds of tokens.
%token
    test  { TokenTest }
    world  { TokenWorld }
    agent  { TokenAgent }
    action  { TokenAction }
    assert  { TokenAssert }
    query  { TokenQuery }
    id  { TokenId }
    int { TokenInt $$ }
    symbol { TokenSym $$ }
    arrow  { TokenArrow }
    '=' { TokenEq }
    '+' { TokenPlus }
    '&' { TokenIntersection }
    '-' { TokenMinus }
    '~' { TokenComplement }
    '*' { TokenStar }
    ';' { TokenProduct }
    '(' { TokenLParen }
    ')' { TokenRParen }
    '[' { TokenLSquareBracket }
    ']' { TokenRSquareBracket }
    '<' { TokenLAngleBracket }
    '>' { TokenRAngleBracket }
-- Above, symbol covers events, tests, and agent names.
      
-- Operator precedences.
%right in
%nonassoc '>' '<'
%left '+' '-' '&'
%left '*' '/'
%left NEG

%%
-- Productions, with constructors for abstract syntax in curly brackets.
-- The root symbol is the LHS of the first production


Root : World Assert Actions Agents Query               { ($1,$2,$3,$4,$5)::([[Char]],Prop,[ActionSpec],[AgentSpec],Prop) }

Assert : assert Prop      { $2 }

-- Boolean generators
Generator : symbol              { [$1] }
    | symbol '+' Generator      { $1 : $3 }

World : world '=' Generator    { $3 }

-- Propositions
Prop : Prop '+' Prop           { Union $1 $3 }
    | Prop ';' id ';' Prop     { Product (Product $1 (Ident "Ev")) $5 }
    | Prop '&' Prop            { Intersection $1 $3 }
    | Prop '-' Prop            { Minus $1 $3 }
    | Prop '*'                 { Star $1 }
    | Prop ';' Prop            { Product $1 $3 }
    | '(' Prop ')'             { $2 }
    | symbol '(' Prop ')'      { Dia $1 $3 }
    | '[' symbol ']' Prop      { Box $2 $4 }
    | '<' symbol '>' Prop      { Dia $2 $4 }    
    | '~' Prop %prec NEG       { Complement $2 }
    | int                      { Int $1 }
    | test symbol              { Test $2 }
    | symbol                   { Ident $1 }

-- Query
Query : query Prop             { $2 }

-- Agents and alternatives

Event : symbol arrow Prop    { EventSpec $1 $3 }
    | '(' Event ')'          { $2 }

Events : Event             { [$1] }
    | Event Events         { $1 : $2 }

Agent : agent symbol '=' Events  { AgentSpec $2 $4 }
Agents : Agent             { [$1] }
     | Agent Agents        { $1 : $2 }

Actions : Action             { [$1] }
    | Action Actions         { $1 : $2 }

-- Action: action symbol '=' Prop ';' id ';' Prop { ActionSpec $2 $4 $8}
Action: action symbol '=' Prop { ActionSpec $2 $4}

{

parseError :: [Token] -> a
parseError _ = error "Parse error"

data Prop = Union Prop Prop
         | Intersection Prop Prop
         | Minus Prop Prop
         | Star Prop
         | Product Prop Prop
         | Complement Prop
         | Brack Prop
         | Int Int
         | Ident String
         | Test String
         | Dia String Prop
         | Box String Prop
         deriving Show

data EventSpec = EventSpec String Prop
     deriving Show

-- data ActionSpec = ActionSpec String Prop Prop
data ActionSpec = ActionSpec String Prop
     deriving Show

data AgentSpec = AgentSpec String [EventSpec]
     deriving Show

parse :: String -> ([[Char]],Prop,[ActionSpec],[AgentSpec],Prop)
parse = parseEpik . scanTokens


goo :: ([[Char]],Prop,[ActionSpec],[AgentSpec],Prop) -> [String]
goo (a,b,c,d,e) = fludefs a

moo :: ([[Char]],Prop,[ActionSpec],[AgentSpec],Prop) -> String
moo (a,b,c,d,e) = statedef0 b

hoo :: ([[Char]],Prop,[ActionSpec],[AgentSpec],Prop) -> [String]
hoo (a,b,c,d,e) = (fludefs a) ++ (statedefs b) ++
                  (map flu_redef a) ++ [(badpairdef a)] ++
                  (eventdefs c) ++ ["source kat.fst"] ++
                  (agentspecs2fst d) ++
                  ["echo " ++ prop2fst(e)] ++
                  ["regex " ++ prop2fst(e) ++ ";", "set print-space ON", "print random-words"]
                  

escapescore:: String -> String
escapescore y = replace "_" "%_" y

main = do args <- getArgs
          let kfile = (head args)
          s <- readFile kfile
          let s' = unlines(hoo(parse s))
          putStrLn s'

flu0 :: Int -> Int -> [String]
flu0 n m
 | m == 1 =  "1" : (replicate (n-1) "Bool")
 | n > 1 = "Bool" : (flu0 (n-1) (m-1))
 | n == 1 = ["1"]
 | n == 0 = []

as_string :: [String] -> String
as_string [x] = x
as_string [] = ""
as_string (x : xs) = x ++ " " ++ (as_string xs)

-- Fluent coded as Fst, with "Bool" in variable positions
-- flu1 6 4 => "[Bool Bool Bool 1 Bool Bool]"
flu1 :: Int -> Int -> String
flu1 n m = "[" ++ (as_string (flu0 n m)) ++ "]"


fludef0 :: String -> Int -> Int -> String
fludef0 x n m = ("define " ++ x ++ " " ++ (flu1 n m) ++ ";") 

-- n is the length of the Boolean vector, it is held constant here
fludefs0 :: [String] -> Int -> Int -> [String]
fludefs0 [] n m = []
fludefs0 (x:xs) n m = (fludef0 x n m) : (fludefs0 xs n (m+1))

fludefs :: [String] -> [String]
fludefs xs = "define Bool [\"0\"|\"1\"];" :
   ("define St0 [" ++ (as_string (replicate (length xs) "Bool")) ++ "];") :
   (fludefs0 xs (length xs) 1)

flu_redef :: String -> String
flu_redef x = (unwords ["define",x,"St &",x]) ++ ";"


statedef0 :: Prop -> String
statedef0 (Intersection p q) = "(" ++ (statedef0 p) ++ " & " ++ (statedef0 q) ++ ")"
statedef0 (Union p q) = "(" ++ (statedef0 p) ++ " | " ++ (statedef0 q) ++ ")"
statedef0 (Ident x) = x
statedef0 (Complement x) = "(St0 - " ++ (statedef0 x) ++ ")"

statedefs :: Prop -> [String]
statedefs p = ["define St St0 & " ++ (statedef0 p) ++ ";",
               "define Nst(X) [St - X];"]

event_name :: ActionSpec -> String
event_name (ActionSpec n p) = (escapescore n)

eventdef :: ActionSpec -> String
eventdef (ActionSpec n p) = unwords ["define",(escapescore n),(eventprop p)] ++ ";"

eventdefs :: [ActionSpec] -> [String]
eventdefs xs = (map (eventdef . substitute_ev) xs) ++
               ["define Event [" ++ (replace " " " | " (unwords (map event_name xs))) ++ "];"]
-- The above using string substitution is crude.

substitute_ev0 :: String -> Prop -> Prop
substitute_ev0 x (Product p q) = (Product (substitute_ev0 x p) (substitute_ev0 x q))
substitute_ev0 x (Union p q) = (Union (substitute_ev0 x p) (substitute_ev0 x q))
substitute_ev0 x (Intersection p q) = (Intersection (substitute_ev0 x p) (substitute_ev0 x q))
substitute_ev0 x (Ident "Ev") = (Ident x)
substitute_ev0 y (Ident z) = (Ident z)
substitute_ev0 y (Test z) = (Test z)

substitute_ev :: ActionSpec -> ActionSpec
substitute_ev (ActionSpec x p) = (ActionSpec x (substitute_ev0 x p))

bracket :: String -> String
bracket y = "[" ++ y ++ "]"

badpair0 :: String -> String
badpair0 x = let nst y = "Nst(" ++ y ++ ")"
              in bracket(unwords [x,(nst x)]) ++ " | " ++ bracket(unwords [(nst x),x])

badpair1 :: [String] -> String
badpair1 [x] = badpair0 x
badpair1 (x:xs) = (badpair0 x) ++ " | " ++ (badpair1 xs)

badpairdef :: [String] -> String
badpairdef xs = "define UnequalStPair " ++ (badpair1 xs) ++ ";"


-- "[[" ++ (escapescore x) ++ " " ++ (escapescore x) ++ "] | [Nst(" ++ x ++ ") Nst(" ++ x ++ ")]]"

eventprop :: Prop -> String
eventprop (Ident x) = (escapescore x)
eventprop (Test x) = x
eventprop (Product x y) = "[" ++ (eventprop x) ++ " " ++ (eventprop y) ++ "]"
eventprop (Union x y) = "[" ++ (eventprop x) ++ " | " ++ (eventprop y) ++ "]"
eventprop (Intersection x y) = "[" ++ (eventprop x) ++ " & " ++ (eventprop y) ++ "]"

-- Consider whether the Event .x. Event part could be elided from the relation composition.
eventspec2fst :: EventSpec -> String
eventspec2fst (EventSpec x p) = "[" ++ (escapescore x) ++ " .o. [Event .x. Event] .o. " ++ (eventprop p) ++ "]"

eventspecs2fst :: [EventSpec] -> String
eventspecs2fst [e] = (eventspec2fst e)
eventspecs2fst (e : es) = "[" ++ (eventspec2fst e) ++ " | " ++ (eventspecs2fst es) ++ "]" 

-- Use an agent name such as "bob" as a name for the corresponding world alternative relation.
-- This includes the relational closure, and so skips defining an event alternative relation.

agentspec2fst :: AgentSpec -> String
agentspec2fst (AgentSpec a es) = "define " ++ a ++ " RelKst(" ++ (eventspecs2fst es) ++ ");"

-- Map a list of AgentSpec to a list of fst definitions (list of strings)
agentspecs2fst :: [AgentSpec] -> [String]
agentspecs2fst [x] = [agentspec2fst x]
agentspecs2fst (x:xs) = (agentspec2fst x) : (agentspecs2fst xs)

-- Use this to map full propositions, the test items.
prop2fst :: Prop -> String
prop2fst (Union p q) = bracket((prop2fst p) ++ " | " ++ (prop2fst q))
prop2fst (Intersection p q) = bracket((prop2fst p) ++ " & " ++ (prop2fst q))
prop2fst (Minus p q) = bracket((prop2fst p) ++ " - " ++ (prop2fst q))
prop2fst (Star p) =  "Kst(" ++ (prop2fst p) ++ ")"
prop2fst (Product p q) = "Cn(" ++ (prop2fst p) ++ "," ++ (prop2fst q) ++ ")"
prop2fst (Complement p) =  "Not(" ++ (prop2fst p) ++ ")"
prop2fst (Dia x p) = "Dia(" ++ x ++ "," ++ (prop2fst p) ++ ")"
prop2fst (Box x p) = "Box(" ++ x ++ "," ++ (prop2fst p) ++ ")"
prop2fst (Test x) = (escapescore x)
prop2fst (Ident x) = (escapescore x)
                                                                   
}

