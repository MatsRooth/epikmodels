import System.IO
import Prelude
import System.Environment

main = do args <- getArgs
          let v = splitBitString (head args)
          let tex = unlines (preamble ++ (n1 v) ++ (n2 v) ++ (n3 v) ++
                             (ea12 v) ++ (ea23 v) ++ (ea31 v) ++
                             (eb12 v) ++ (eb23 v) ++ (eb31 v) ++ postamble)
          putStrLn tex

splitBitString :: String -> [Bool]
splitBitString [] = []
splitBitString ('0':cs) = False : (splitBitString cs)
splitBitString ('1':cs) = True : (splitBitString cs)
splitBitString (' ':cs) = (splitBitString cs)
splitBitString (c:cs) = error ("Character '" ++ (c:"' is not 0, 1 or space."))

-- The following piece together a tex program from lines, using lists of strings.
-- Tex "\" has to be represented as "\\" in a string. 
preamble :: [String]
preamble = ["\\documentclass[border=5pt,tikz]{standalone}",
            "\\usetikzlibrary{positioning,shapes,shapes.geometric}",
            "\\begin{document}",
            "\\begin{tikzpicture}[every node/.style={draw, rounded rectangle,line width=5.0pt,scale=2.0},line width=3.0pt]"]

-- Beginning and end of the tex program.
postamble :: [String]
postamble = ["\\end{tikzpicture}",
             "\\end{document}"]

-- Declarations of three nodes in the graph.
-- One node is marked as the base world, using a hollow dot instead of a solid one.
-- Which one is determined by the first two fluents MB and MA.
-- Node 1 is at 150 degrees.
n1 :: [Bool] -> [String]
n1 v = let nd = "\\node (1) at (150:3cm) "
           sh = (shape (v!!8) (v!!11))
           fi = (fill False True v)
           en = "{};"
       in [nd ++ sh ++ fi ++ en]

-- Node 2 is at 30 degrees.
n2 :: [Bool] -> [String]
n2 v = let nd = "\\node (2) at (30:3cm) "
           sh = (shape (v!!9) (v!!12))
           fi = (fill True False v)
           en = "{};"
       in [nd ++ sh ++ fi ++ en]


-- Node 3 is at 270 degrees.
n3 :: [Bool] -> [String]
n3 v = let nd = "\\node (3) at (270:3cm) "
           sh = (shape (v!!10) (v!!13))
           fi = (fill True True v)
           en = "{};"
       in [nd ++ sh ++ fi ++ en]
          
-- This is the fluent layout in muddy2graph1.k.
-- world = MB + MA + Ea12 + Ea23 + Ea31 + Eb12 + Eb23 + Eb31 + Ra1 + Ra2 + Ra3 + Rb1 + Rb2 + Rb3

-- There are six potential edges, which have a constant shape and can be present or not.
-- Amy edges are solid.
-- Amy edge fluents Ea12 Ea23 and Ea31 are in positions 2,3 and 4.

ea12 v = if (v!!2) then ["\\draw (1) to[bend left=20] (2);"] else []
ea23 v = if (v!!3) then ["\\draw (2) to[bend left=20] (3);"] else []
ea31 v = if (v!!4) then ["\\draw (3) to[bend left=20] (1);"] else []

-- Bob edges are dashed.
-- Bob edge fluents Eb12 Eb23 and Eb31 are in positions 5,6 and 7.

eb12 v = if (v!!5) then ["\\draw [dashed] (1) to[bend right=15] (2);"] else []
eb23 v = if (v!!6) then ["\\draw [dashed] (2) to[bend right=15] (3);"] else []
eb31 v = if (v!!7) then ["\\draw [dashed] (3) to[bend right=15] (1);"] else []


-- The node center is white if the node world is the base world, and is gray otherwise.
-- The base world is identified by MB and MA in v.
-- The node world is identified by the node function via a pair of Booleans.
fill :: Bool -> Bool -> [Bool] -> String
fill mb ma v = if (v!!0 == mb && v!!1 == ma) then ",fill=white]" else ",fill=gray]"


-- There are node 2x2 shapes, which in a node i are determined by the fluents Rai and Rbi.
-- These can be selected by the node function.
-- The node shape is independent of the base world.

shape :: Bool -> Bool -> String
shape False False = "[rounded rectangle"
shape True False = "[rounded rectangle west arc=none, rotate=-90"
shape False True = "[rounded rectangle east arc=none, rotate=-90"
shape True True = "[rectangle"

