"# Lexer-Parser-Interpreter" 

Python version used: 3.9

Sample use: run Parser_custom.py's __main__
it should read program.txt, do lexical analysis, tokenize, parse, draw parse tree to parse_tree.xml and interpret.
In the program.txt is a gcd finder... You will be asked in console for 2 inputs for which gcd should be found.

Grammar used is in pam.g4 file. It is extended according to task 1.2. (added AND, OR, NOT); called extended PAM grammar. Additional rules may be specified in 2.2. and 1.3. tasks, see pdfs.
Basically for 1.3. I needed the parse_tree.xml file generated, while for 2.2. I needed the interpreter of the program.

Inspired by https://ruslanspivak.com/lsbasi-part10/ to build specifically this architecture. 
Took part 10 code and adapted.