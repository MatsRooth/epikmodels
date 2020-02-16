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
  query                         { \s -> TokenQuery }
  id                            { \s -> TokenId }
  world                         { \s -> TokenWorld }
  \-\>                          { \s -> TokenArrow }
  $digit+                       { \s -> TokenInt (read s) }
  \=                            { \s -> TokenEq }
  \+                            { \s -> TokenPlus }
  \&                            { \s -> TokenIntersection }
  \-                            { \s -> TokenMinus }
  \~                            { \s -> TokenComplement }
  \*                            { \s -> TokenStar }
  \;                            { \s -> TokenProduct }
  \(                            { \s -> TokenLParen }
  \)                            { \s -> TokenRParen }
  $alpha [$alpha $digit \_ \']* { \s -> TokenSym s }

{

-- The token type:
data Token = TokenAssert
           | TokenAgent
           | TokenAction
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
           | TokenComplement
           | TokenStar
           | TokenProduct
           | TokenLParen
           | TokenRParen
           deriving (Eq,Show)

scanTokens = alexScanTokens

}
