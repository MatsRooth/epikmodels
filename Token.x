{
module Token where
}

%wrapper "basic"

$digit = 0-9
$alpha = [a-zA-Z]

tokens :-

  $white+                       ;
  "--".*                        ;
  assert                        { \s -> TokenAssert }
  test                          { \s -> TokenTest }
  agent                         { \s -> TokenAgent }
  action                        { \s -> TokenAction }
  queryall                      { \s -> TokenQueryAll }
  query                         { \s -> TokenQuery }
  id                            { \s -> TokenId }
  world                         { \s -> TokenWorld }
  \-\>                          { \s -> TokenArrow }
  $digit+                       { \s -> TokenInt (read s) }
  \=                            { \s -> TokenEq }
  \+                            { \s -> TokenPlus }
  \&                            { \s -> TokenIntersection }
  \-                            { \s -> TokenMinus }
  \_                            { \s -> TokenUnderscore }
  \~                            { \s -> TokenComplement }
  \*                            { \s -> TokenStar }
  \;                            { \s -> TokenProduct } 
  \[                            { \s -> TokenLSquareBracket }
  \]                            { \s -> TokenRSquareBracket }
  \<                            { \s -> TokenLAngleBracket }
  \>                            { \s -> TokenRAngleBracket }  
  \(                            { \s -> TokenLParen }
  \)                            { \s -> TokenRParen }
  $alpha [$alpha $digit \_ \']* { \s -> TokenSym s }

{

-- The token type:
data Token = TokenAssert
           | TokenAgent
           | TokenAction
           | TokenQueryAll
           | TokenQuery
           | TokenId
           | TokenWorld
           | TokenTest
           | TokenArrow
           | TokenInt Int
           | TokenSym String
           | TokenEq
           | TokenPlus
           | TokenIntersection
           | TokenMinus
           | TokenUnderscore	   
           | TokenComplement
           | TokenStar
           | TokenProduct
           | TokenLParen
           | TokenRParen
           | TokenLSquareBracket
           | TokenRSquareBracket
           | TokenLAngleBracket
           | TokenRAngleBracket
         deriving (Eq,Show)

scanTokens = alexScanTokens

}
