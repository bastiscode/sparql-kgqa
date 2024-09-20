%start iriOrLiteral

%%

iriOrLiteral
    : iri
    | RDFLiteral 
    | NumericLiteral 
    | BooleanLiteral 
    ;

RDFLiteral
    : String
    | String 'LANGTAG'
    | String '^^' iri
    ;


NumericLiteral
    : NumericLiteralUnsigned 
    | NumericLiteralPositive 
    | NumericLiteralNegative
    ;

NumericLiteralUnsigned
    : 'INTEGER' 
    | 'DECIMAL' 
    | 'DOUBLE'
    ;

NumericLiteralPositive
    : 'INTEGER_POSITIVE' 
    | 'DECIMAL_POSITIVE' 
    | 'DOUBLE_POSITIVE'
    ;

NumericLiteralNegative
    : 'INTEGER_NEGATIVE' 
    | 'DECIMAL_NEGATIVE' 
    | 'DOUBLE_NEGATIVE'
    ;

BooleanLiteral
    : 'true' 
    | 'false'
    ;

String
    : 'STRING_LITERAL1' 
    | 'STRING_LITERAL2' 
    | 'STRING_LITERAL_LONG1' 
    | 'STRING_LITERAL_LONG2'
    ;

iri
    : 'IRIREF' 
    | PrefixedName
    ;

PrefixedName
    : 'PNAME_LN' 
    | 'PNAME_NS'
    ;
