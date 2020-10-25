import Syntax

main = do let out1 = unlines (testDef b0 phi0)
          putStrLn out1
          let out2  = unlines (effectDef evspec)
          putStrLn out2
          let out3  = unlines (agentDef agentspec)
          putStrLn out3

