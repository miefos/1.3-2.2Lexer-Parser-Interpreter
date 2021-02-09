DRAW_TREE = True
DRAW_TO = './parse_tree.xml'  # parse tree xml file
f = open(DRAW_TO, "w")


def print_w_tabs(deep, text):
    if not DRAW_TREE:
        return
    i = 0
    for i in range(deep):
        f.write("\t")
    f.write(str(text) + "\n")

###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################
INTEGER_CONST = 'INTEGER_CONST'
REAL_CONST    = 'REAL_CONST'
PLUS          = 'PLUS'
MINUS         = 'MINUS'
MUL           = 'MUL'
INTEGER_DIV   = 'INTEGER_DIV'
FLOAT_DIV     = 'FLOAT_DIV'
LPAREN        = 'LPAREN'
RPAREN        = 'RPAREN'
ID            = 'ID'
ASSIGN        = 'ASSIGN'
END           = 'END'
SEMI          = 'SEMI'
PROGRAM       = 'PROGRAM'
COMMA         = 'COMMA'
EOF           = 'EOF'
IF            = 'IF'
THEN          = 'THEN'
ELSE          = 'ELSE'
FI            = 'FI'
WHILE         = 'WHILE'
DO            = 'DO'
READ          = 'READ'
WRITE         = 'WRITE'
AND           = 'AND'
OR            = 'OR'
NOT           = 'NOT'
# Relation
LE            = 'LE'
GE            = 'GE'
GREATER       = 'GREATER'
LESS          = 'LESS'
EQUAL         = 'EQUAL'
NOTEQUAL      = 'NOTEQUAL'
RELATION      = [LE, GE, GREATER, LESS, EQUAL, NOTEQUAL]


class Token(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value

RESERVED_KEYWORDS = {
    'program': Token('PROGRAM', 'PROGRAM'),
    'DIV': Token('INTEGER_DIV', 'DIV'),
    'if': Token('IF', 'IF'),
    'then': Token('THEN', 'THEN'),
    'else': Token('ELSE', 'ELSE'),
    'fi': Token('FI', 'FI'),
    'and': Token('AND', 'AND'),
    'or': Token('OR', 'OR'),
    'not': Token('NOT', 'NOT'),
    'while': Token('WHILE', 'WHILE'),
    'do': Token('DO', 'DO'),
    'end': Token('END', 'END'),
    'read': Token('READ', 'READ'),
    'write': Token('WRITE', 'WRITE'),
}

class Lexer(object):
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos]

    def error(self):
        raise Exception('Invalid character')

    def advance(self):
        """Advance the `pos` pointer and set the `current_char` variable."""
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self):
        """Return a (multidigit) integer or float consumed from the input."""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        if self.current_char == '.':
            result += self.current_char
            self.advance()

            while (
                self.current_char is not None and
                self.current_char.isdigit()
            ):
                result += self.current_char
                self.advance()

            token = Token('REAL_CONST', float(result))
        else:
            token = Token('INTEGER_CONST', int(result))

        return token

    def _id(self):
        """Handle identifiers and reserved keywords"""
        result = ''
        while self.current_char is not None and self.current_char.isalnum():
            result += self.current_char
            self.advance()

        token = RESERVED_KEYWORDS.get(result, Token(ID, result))
        return token

    def get_next_token(self):
        """ Tokenizer """
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isalpha():
                return self._id()

            if self.current_char.isdigit():
                return self.number()

            if self.current_char == ':' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(ASSIGN, ':=')

            if self.current_char == '>' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(GE, '>=')

            if self.current_char == '<' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(LE, '<=')


            if self.current_char == '<' and self.peek() == '>':
                self.advance()
                self.advance()
                return Token(NOTEQUAL, '<>')

            if self.current_char == '<':
                self.advance()
                return Token(LESS, '<')

            if self.current_char == '>':
                self.advance()
                return Token(GREATER, '>')

            if self.current_char == '=':
                self.advance()
                return Token(EQUAL, '=')

            if self.current_char == ';':
                self.advance()
                return Token(SEMI, ';')

            if self.current_char == ',':
                self.advance()
                return Token(COMMA, ',')

            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')

            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')

            if self.current_char == '*':
                self.advance()
                return Token(MUL, '*')

            if self.current_char == '/':
                self.advance()
                return Token(FLOAT_DIV, '/')

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')

            self.error()

        return Token(EOF, None)


###############################################################################
#                                                                             #
#  PARSER                                                                     #
#                                                                             #
###############################################################################

class AST(object):
    pass


class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class UnaryOp(AST):
    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr


class Assign(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class IfThenElse(AST):
    def __init__(self, comparison, thenSeries, elseSeries):
        self.comparison = comparison
        self.thenSeries = thenSeries
        self.elseSeries = elseSeries  # sometimes NoOp


class Varlist(AST):
    def __init__(self):
        self.vars = []


class Var(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class Compare(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right
        self.andOrCompare: [Compare, Token] = None


class While(AST):
    def __init__(self, variant, doSeries):
        self.variant = variant
        self.doSeries = doSeries


class NoOp(AST):
    pass


class Program(AST):
    def __init__(self, series):
        self.series = series


class Series(AST):
    def __init__(self):
        self.stmts = []


class InputStmt(AST):
    def __init__(self):
        self.varlist = Varlist()


class OutputStmt(AST):
    def __init__(self):
        self.varlist = Varlist()


class VarDecl(AST):
    def __init__(self, var_node, type_node):
        self.var_node = var_node
        self.type_node = type_node


class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        # set current token to the first token taken from the input
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('Invalid syntax', self.current_token.value)

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def program(self):
        self.eat(PROGRAM)
        series_node = self.series()
        program_node = Program(series_node)
        return program_node

    def series(self):
        node = self.statement()

        results_node = Series()
        results_node.stmts = [node]

        while self.current_token.type == SEMI:
            self.eat(SEMI)
            results_node.stmts.append(self.statement())

        return results_node

    def statement(self):
        if self.current_token.type == ID:
            node = self.assignment_statement()
        elif self.current_token.type == READ:
            node = self.input_stmt()
        elif self.current_token.type == WRITE:
            node = self.output_stmt()
        elif self.current_token.type == IF:
            node = self.if_stmt()
        elif self.current_token.type == WHILE:
            node = self.while_stmt()
        else:
            node = self.empty()
        return node

    def input_stmt(self):
        self.eat(READ)
        node = InputStmt()
        node.varlist = self.varlist()
        return node

    def output_stmt(self):
        self.eat(WRITE)
        node = OutputStmt()
        node.varlist = self.varlist()
        return node

    def if_stmt(self):
        self.eat(IF)
        compare = self.compar()
        self.eat(THEN)
        thenSeries = self.series()
        if self.current_token.type == ELSE:
            self.eat(ELSE)
            elseSeries = self.series()
        else:
            elseSeries = Series()
            elseSeries.stmts.append(NoOp)
        self.eat(FI)

        node = IfThenElse(compare, thenSeries, elseSeries)
        return node

    def compar(self):
        left = self.expr()
        relation = self.current_token
        if relation.type in RELATION:
            self.eat(self.current_token.type)
        else:
            raise Exception('invalid relation')
        right = self.expr()

        node = Compare(left, relation, right)

        while self.current_token.type in ['AND', 'OR']:
            andOrToken = self.current_token
            if self.current_token.type == 'AND':
                self.eat(AND)
            else:
                self.eat(OR)

            left = self.expr()
            relation = self.current_token
            if relation.type in RELATION:
                self.eat(self.current_token.type)
            else:
                raise Exception('invalid relation')
            right = self.expr()

            node_tmp = Compare(left, relation, right)
            node_tmp.andOrCompare = [node, andOrToken]
            node = node_tmp
        return node

    def while_stmt(self):
        self.eat(WHILE)
        compare = self.compar()
        self.eat(DO)
        doSeries = self.series()
        self.eat(END)

        return While(compare, doSeries)

    def varlist(self):
        resultVarList: Varlist = Varlist()

        node = Var(self.current_token)
        self.eat(ID)
        resultVarList.vars.append(node)

        while self.current_token.type == COMMA:
            self.eat(COMMA)
            node = Var(self.current_token)
            self.eat(ID)
            resultVarList.vars.append(node)

        return resultVarList

    def assignment_statement(self):
        left = self.variable()
        token = self.current_token
        self.eat(ASSIGN)
        right = self.expr()
        node = Assign(left, token, right)
        return node

    def variable(self):
        node = Var(self.current_token)
        self.eat(ID)
        return node

    def empty(self):
        return NoOp()

    def expr(self):
        node = self.term()

        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
            elif token.type == MINUS:
                self.eat(MINUS)

            node = BinOp(left=node, op=token, right=self.term())

        return node

    def term(self):
        node = self.factor()

        while self.current_token.type in (MUL, INTEGER_DIV, FLOAT_DIV):
            token = self.current_token
            if token.type == MUL:
                self.eat(MUL)
            elif token.type == INTEGER_DIV:
                self.eat(INTEGER_DIV)
            elif token.type == FLOAT_DIV:
                self.eat(FLOAT_DIV)

            node = BinOp(left=node, op=token, right=self.factor())

        return node

    def factor(self):
        token = self.current_token
        if token.type == PLUS:
            self.eat(PLUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == MINUS:
            self.eat(MINUS)
            node = UnaryOp(token, self.factor())
            return node
        if token.type == NOT:
            self.eat(NOT)
            node = UnaryOp(token, self.expr())
            return node
        elif token.type == INTEGER_CONST:
            self.eat(INTEGER_CONST)
            return Num(token)
        elif token.type == REAL_CONST:
            self.eat(REAL_CONST)
            return Num(token)
        elif token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            self.eat(RPAREN)
            return node
        else:
            node = self.variable()
            return node

    def parse(self):
        node = self.program()
        if self.current_token.type != EOF:
            self.error()

        return node


###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################

class NodeVisitor(object):
    def visit(self, node, deep = None):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        if deep is not None:
            return visitor(node, deep)
        else:
            return visitor(node)

    def generic_visit(self, node, deep=False):
        raise Exception('No visit_{} method, {}, {}'.format(type(node).__name__, type(node), node),)


class Interpreter(NodeVisitor):
    def __init__(self, parser):
        self.parser = parser
        self.GLOBAL_SCOPE = {}

    def convert_to_number(self, user_input):
        try:
            return int(user_input)
        except ValueError:
            try:
                return float(user_input)
            except ValueError:
                print("Error. Entered invalid number.")

    def visit_Program(self, node: Program):
        self.visit(node.series)

    def visit_Series(self, node: Series):
        for stmt in node.stmts:
            self.visit(stmt)

    def visit_While(self, node: While):
        while self.visit(node.variant):
            self.visit(node.doSeries)

    def visit_Assign(self, node: Assign):
        var_name = node.left.value
        self.GLOBAL_SCOPE[var_name] = self.visit(node.right)

    def visit_InputStmt(self, node: InputStmt):
        vars = self.visit(node.varlist)
        for var in vars:
            received_input = input()
            received_input = self.convert_to_number(received_input)
            self.GLOBAL_SCOPE[var.value] = received_input

    def visit_IfThenElse(self, node: IfThenElse):
        isTrue = self.visit(node.comparison)
        if isTrue:
            self.visit(node.thenSeries)
        else:
            self.visit(node.elseSeries)

    def visit_Compare(self, node: Compare):
        result = False
        result1 = False
        result2 = False
        if node.andOrCompare is not None:
            result2 = self.visit(node.andOrCompare[0])

        if node.op.type == LESS:
            if self.visit(node.left) < self.visit(node.right):
                result1 = True
            else:
                result1 = False
        elif node.op.type == GREATER:
            if self.visit(node.left) > self.visit(node.right):
                result1 = True
            else:
                result1 = False
        elif node.op.type == LE:
            if self.visit(node.left) <= self.visit(node.right):
                result1 = True
            else:
                result1 = False
        elif node.op.type == GE:
            if self.visit(node.left) >= self.visit(node.right):
                result1 = True
            else:
                result1 = False
        elif node.op.type == EQUAL:
            if self.visit(node.left) == self.visit(node.right):
                result1 = True
            else:
                result1 = False
        elif node.op.type == NOTEQUAL:
            if not self.visit(node.left) == self.visit(node.right):
                result1 = True
            else:
                result1 = False

        if node.andOrCompare is not None:
            if node.andOrCompare[1].type == AND:
                result = (result1 and result2)
            elif node.andOrCompare[1].type == OR:
                result = (result1 or result2)
        else:
            result = result1

        return result

    def visit_OutputStmt(self, node: OutputStmt):
        vars_ = self.visit(node.varlist)
        for var in vars_:
            print(self.GLOBAL_SCOPE.get(var.value))

    def visit_Varlist(self, node: Varlist):
        return node.vars

    def visit_Var(self, node: Var):
        var_name = node.value
        var_value = self.GLOBAL_SCOPE.get(var_name)
        if var_value is None:
            raise NameError(repr(var_name))
        else:
            return var_value

    def visit_Token(self, node: Token):
        pass

    def visit_Num(self, node: Num):
        return node.value

    def visit_BinOp(self, node: BinOp):
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == INTEGER_DIV:
            return self.visit(node.left) // self.visit(node.right)
        elif node.op.type == FLOAT_DIV:
            return float(self.visit(node.left)) / float(self.visit(node.right))

    def visit_UnaryOp(self, node: UnaryOp):
        op = node.op.type
        if op == PLUS:
            return +self.visit(node.expr)
        elif op == MINUS:
            return -self.visit(node.expr)
        elif op == NOT:
            return not self.visit(node.expr)

    def visit_NoOp(self, node):
        pass

    def interpret(self, tree):
        if tree is None:
            return ''
        return self.visit(tree)

###############################################################################
#                                                                             #
#  AST Drawer                                                                #
#                                                                             #
###############################################################################

class ASTDrawer(NodeVisitor):
    def __init__(self, parser):
        self.parser = parser

    def visit_Program(self, node: Program, deep):
        print_w_tabs(deep, "<program>")
        self.visit(node.series, deep+1)
        print_w_tabs(deep, "</program>")

    def visit_Series(self, node: Series, deep: int):
        for stmt in node.stmts:
            self.visit(stmt, deep)

    def visit_While(self, node: While, deep):
        print_w_tabs(deep, "<while>")
        self.visit(node.variant, deep+1)
        print_w_tabs(deep, "<do>")
        self.visit(node.doSeries, deep+1)
        print_w_tabs(deep, "<end>")

    def visit_Assign(self, node: Assign, deep):
        print_w_tabs(deep, "<assign>")
        self.visit(node.left, deep+1)
        self.visit(node.op, deep+1)
        self.visit(node.right, deep+1)
        print_w_tabs(deep, "</assign>")

    def visit_InputStmt(self, node: InputStmt, deep):
        print_w_tabs(deep, "<read>")
        self.visit(node.varlist, deep+1)
        print_w_tabs(deep, "</read>")

    def visit_IfThenElse(self, node: IfThenElse, deep):
        print_w_tabs(deep, "<if>")
        self.visit(node.comparison, deep+1)
        print_w_tabs(deep, "<then>")
        self.visit(node.thenSeries, deep+1)
        print_w_tabs(deep, "<else>")
        self.visit(node.elseSeries, deep+1)
        print_w_tabs(deep, "<fi>")

    def visit_Compare(self, node: Compare, deep):
        self.visit(node.left, deep)
        self.visit(node.op, deep)
        self.visit(node.right, deep)
        if node.andOrCompare is not None:
            print_w_tabs(deep+1, "<" + node.andOrCompare[1].type + ">")
            self.visit(node.andOrCompare[0], deep+1)
            print_w_tabs(deep+1, "</" + node.andOrCompare[1].type + ">")

    def visit_OutputStmt(self, node: OutputStmt, deep):
        print_w_tabs(deep, "<write>")
        self.visit(node.varlist, deep+1)
        print_w_tabs(deep, "</write>")

    def visit_Varlist(self, node: Varlist, deep):
        for var in node.vars:
            self.visit(var, deep)

    def visit_Var(self, node: Var, deep):
        print_w_tabs(deep, node.value)

    def visit_Token(self, node: Token, deep):
        print_w_tabs(deep, node.value)

    def visit_Num(self, node: Num, deep):
        print_w_tabs(deep, node.value)

    def visit_BinOp(self, node: BinOp, deep):
        print_w_tabs(deep, "<BinOp>")
        self.visit(node.left, deep+1)
        self.visit(node.op, deep+1)
        self.visit(node.right, deep+1)
        print_w_tabs(deep, "</BinOp>")

    def visit_UnaryOp(self, node: UnaryOp, deep):
        print_w_tabs(deep, "<UnaryOp>")
        self.visit(node.op, deep+1)
        self.visit(node.expr, deep + 1)
        print_w_tabs(deep, "</UnaryOp>")

    def visit_NoOp(self, node, deep):
        pass

    def draw(self, tree):
        if tree is None:
            return ''
        deep = 0
        return self.visit(tree, deep)


def main():
    text = open('Pascal.ps', 'r').read()

    lexer = Lexer(text)
    parser = Parser(lexer)
    interpreter = Interpreter(parser)

    tree = parser.parse()

    ASTDrawer(parser).draw(tree)
    interpreter.interpret(tree)


if __name__ == '__main__':
    main()
    f.close()