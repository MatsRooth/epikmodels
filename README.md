# epikmodels
Epik is a language for defining computable multi-agent Kripke frames based on
Kleene Algebra with tests, construction of worlds as guarded strings in a guarded string model for KAT,
construction of propositions as regular sets of guarded strings, epistemic modeling using event 
alternatives as in dynamic epistemic logic, and construction of epistemic alternative relations
as regular relations between guarded strings.

See https://github.com/ericthewry/epik.

This repository contains Epik example models, Haskell code for translating Epik programs to Fst,
and the embedding of KAT in Fst.

## Publication references
Campbell, Eric Hayden, and Mats Rooth. "Epistemic semantics in guarded string models." In Proceedings of the Society for Computation in Linguistics 2021, pp. 81-90. 2021.

Rooth, Mats. "Finite state intensional semantics." In Proceedings of the 12th International Conference on Computational Semantics (IWCS)—Long papers. 2017.

## In progress
Building Epik models using the HFST Python API.  It replaces the Haskell code from Cambell & Rooth that 
syntactically constructed FST programs.

A probabilistic version of Epik, where informations states are probability distributions on sets of guarded
strings, rather than mere propositions.


Languages: Haskell, Python, Fst (hfst/xfst/foma).
