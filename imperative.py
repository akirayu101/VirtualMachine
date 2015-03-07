__author__ = 'hzyuxin'

import operator
import logging
from functools import partial


class Expression(object):

    def __init__(self, vm):
        self.vm = vm
        self.offset = 0

    def codegen(self):
        pass

    def set_cur_offset(self, offset):
        self.offset = offset

    def codegen_expression(self, e):
        if isinstance(e, Expression):
            return e.codegen()
        elif isinstance(e, int):
            return ['loadc ' + str(e)]
        elif isinstance(e, str):
            return ['loada ' + str(self.vm.address(e))]

    def push_code(self):
        self.vm.C.extend(self.codegen())

    def push_sub_code(self, sub_code):
        self.vm.C.extend(sub_code)

    @staticmethod
    def replace_break_continue(statements, break_address, continue_address):
        for i, statement in enumerate(statements):
            if statement == 'break':
                statements[i] = 'jump ' + str(break_address)
            elif statement == 'continue':
                statements[i] = 'jump ' + str(continue_address)
            else:
                pass

        return statements


class Statement(Expression):

    def __init__(self, exp):
        super(Statement, self).__init__(None)
        self.exp = exp

    def codegen(self):
        self.exp.set_cur_offset(self.offset)
        return self.exp.codegen() + ['pop']


class BreakStatement(Statement):

    def __init__(self):
        super(BreakStatement, self).__init__(None)

    def codegen(self):
        return ['break']


class ContinueStatement(Statement):

    def __init__(self):
        super(ContinueStatement, self).__init__(None)

    def codegen(self):
        return ['continue']


class PrintStatement(Statement):
    def __init__(self, name):
        super(PrintStatement, self).__init__(None)
        self.name = name

    def codegen(self):
        return ['show ' + self.name]


# block statements, include if_statement, while_statement and for_statement
# 1.all block statements require jump and jumpz for implementation
# 2.jumpz need a current block position for right jump to solve nested blocks
# if(e1) s1 else s2, s1 and s2 are list of statements
class IfStatement(Expression):

    def __init__(self, vm, e1, s1, s2):
        super(IfStatement, self).__init__(vm)
        self.e1 = e1
        self.s1 = s1
        self.s2 = s2

    def codegen(self):
        '''
        code e1
        jumpz A
        code s1
        jumpz B
        A:
        code s2
        B:...
        '''
        codegen_e1 = self.codegen_expression(self.e1)

        codegen_s1 = []
        for statement in self.s1:
            offset = self.offset + len(codegen_s1) + 1 + len(codegen_e1)
            statement.set_cur_offset(offset)
            codegen_s1.extend(statement.codegen())

        codegen_s2 = []
        for statement in self.s2:
            offset = self.offset + len(codegen_s1) + 2 + len(codegen_s2) + len(codegen_e1)
            statement.set_cur_offset(offset)
            codegen_s2.extend(statement.codegen())

        jump_distance_s1 = self.vm.get_program_len() + len(codegen_e1) + \
            len(codegen_s1) + 2 + self.offset
        jump_distance_s2 = self.vm.get_program_len() + len(codegen_e1) + \
            len(codegen_s1) + len(codegen_s2) + 2 + self.offset

        return codegen_e1 + ['jumpz ' + str(jump_distance_s1)] + codegen_s1 + [
            'jump ' + str(jump_distance_s2)] + codegen_s2


# while(e1) s1
class WhileStatement(Expression):

    def __init__(self, vm, e1, s1):
        super(WhileStatement, self).__init__(vm)
        self.e1 = e1
        self.s1 = s1

    def codegen(self):
        '''
        A:
        code e1
        jumpz B
        code s1
        jump A
        B: ...
        '''
        codegen_e1 = self.codegen_expression(self.e1)

        codegen_s1 = []
        for statement in self.s1:
            offset = self.offset + len(codegen_s1) + 1 + len(codegen_e1)
            statement.set_cur_offset(offset)
            codegen_s1.extend(statement.codegen())

        jump_distance_s1 = self.vm.get_program_len() + len(codegen_e1) + \
            len(codegen_s1) + 2 + self.offset

        break_address = jump_distance_s1
        continue_address = self.vm.get_program_len() + self.offset

        return Expression.replace_break_continue(codegen_e1 + ['jumpz ' + str(jump_distance_s1)] + codegen_s1 + ['jump ' + str(self.vm.get_program_len() + self.offset)], break_address, continue_address)


# for(e1;e2;e3) s1
class ForStatement(Expression):

    def __init__(self, vm, e1, e2, e3, s1):
        super(ForStatement, self).__init__(vm)
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        self.s1 = s1

    def codegen(self):
        '''
        code e1
        pop
        A:
        code e2
        jumpz B
        code s1
        code e3
        pop
        jump A
        B:
        '''
        codegen_e1 = self.codegen_expression(self.e1)
        codegen_e2 = self.codegen_expression(self.e2)
        codegen_e3 = self.codegen_expression(self.e3)

        codegen_s1 = []
        for statement in self.s1:
            offset = self.offset + len(codegen_s1) + 2 + len(codegen_e1) + len(codegen_e2)
            statement.set_cur_offset(offset)
            codegen_s1.extend(statement.codegen())

        jump_distance_a = self.vm.get_program_len() + len(codegen_e1) + 1 + self.offset
        jump_distance_b = self.vm.get_program_len() + len(codegen_e1) + \
            len(codegen_e2) + len(codegen_e3) + len(codegen_s1) + 4 + self.offset

        break_address = jump_distance_b
        continue_address = jump_distance_a

        return Expression.replace_break_continue(codegen_e1 + ['pop'] + codegen_e2 + ['jumpz ' + str(jump_distance_b)] + codegen_s1 + codegen_e3 + ['pop'] + ['jump ' + str(jump_distance_a)], break_address, continue_address)


class BinaryExpression(Expression):
    op_dict = {
        '+': 'add',
        '-': 'sub',
        '/': 'div',
        '%': 'mod',
        '*': 'mul',
        '&&': 'and',
        '!=': 'neq',
        '<=': 'leq',
        '<': 'le',
        '>': 'gr',
        '>=': 'geq',
        '||': 'or',
    }

    def __init__(self, vm, left, right, op):
        super(BinaryExpression, self).__init__(vm)
        self.left = left
        self.right = right
        self.op = op

    def codegen(self):
        codegen_l = self.codegen_expression(self.left)
        codegen_r = self.codegen_expression(self.right)

        return codegen_l + codegen_r + [BinaryExpression.op_dict[self.op]]


class AssignmentExpression(Expression):

    def __init__(self, vm, left, right):
        super(AssignmentExpression, self).__init__(vm)
        self.left = left
        self.right = right

    def codegen(self):
        address = self.vm.address(self.left)
        codegen_l = ['loadc ' + str(address)]
        codegen_r = self.codegen_expression(self.right)
        return codegen_r + codegen_l + ['store']


class IR(object):

    def __init__(self, vm):
        self.vm = vm

        self.binary_ops = {
            'add': operator.add,
            'sub': operator.sub,
            'div': operator.div,
            'mod': operator.mod,
            'mul': operator.mul,
            'and': operator.iand,
            'neq': lambda a, b: a != b,
            'leq': operator.le,
            'geq': operator.ge,
            'gr': operator.gt,
            'eq': operator.eq,
            'le': operator.lt,
            'or': operator.ior,
        }

        self.unary_ops = {
            'neg': operator.neg,
            'not': operator.not_,
        }

        self.init_binary_ops()
        self.init_unary_ops()

    def loadc(self, q):
        self.vm.stack_push(q)

    def load(self):
        address = self.vm.stack_pop()
        self.vm.stack_push(self.vm.S[address])

    def loada(self, address):
        self.vm.stack_push(self.vm.S[address])

    def store(self):
        address = self.vm.stack_pop()
        value = self.vm.stack_top()
        self.vm.S[address] = value

    def storea(self, address):
        value = self.vm.stack_top()
        self.vm.S[address] = value

    def jump(self, address):
        self.vm.PC = address

    def jumpz(self, address):
        value = self.vm.stack_pop()
        if value in [0, False]:
            self.vm.PC = address

    def show(self, variable_name):
        logging.warn('show variable name %s[%d]' % (variable_name, self.vm.inspect(variable_name)))

    def pop(self):
        self.vm.stack_pop()

    def dup(self):
        self.vm.stack_push(self.vm.stack_top())

    def alloc(self, name):
        self.vm.stack_frame.assign(name, None)

    def binary_op(self, op_name):
        r = self.vm.stack_pop()
        l = self.vm.stack_pop()
        lam = self.binary_ops[op_name]
        self.vm.stack_push(lam(l, r))

    def unary_op(self, op_name):
        l = self.vm.stack_pop()
        lam = self.unary_ops[op_name]
        self.vm.stack_push(lam(l))

    def init_binary_ops(self):
        for k in self.binary_ops.iterkeys():
            fn = partial(self.binary_op, k)
            setattr(self, k, fn)

    def init_unary_ops(self):
        for k in self.unary_ops.iterkeys():
            fn = partial(self.unary_op, k)
            setattr(self, k, fn)

    def execute_ir(self, ir_exp):
        ir_exp = ir_exp.split()
        ir_name = ir_exp[0]
        try:
            ir_args = [int(i) for i in ir_exp[1:]]
        except ValueError:
            ir_args = ir_exp[1:]
        getattr(self, ir_name)(*ir_args)


class VM(object):

    def __init__(self):
        self.S = []  # main memory
        self.C = []  # program ir stack
        self.PC = 0  # program counter
        self.SP = -1
        self.IR = IR(self)
        self.stack_frame = StackFrame(self, None)

    def run(self):
        while self.PC != len(self.C):
            ir_exp = self.C[self.PC]
            self.PC += 1
            self.IR.execute_ir(ir_exp)

    def address(self, name):
        return self.stack_frame.address(name)

    def static_assign(self, name, value):
        self.stack_frame.assign(name, value)

    def inspect(self, name):
        address = self.address(name)
        return self.S[address]

    def get_program_len(self):
        return len(self.C)

    def stack_pop(self):
        self.SP -= 1
        return self.S[self.SP + 1]

    def stack_push(self, value):
        if len(self.S) == self.SP + 1:
            self.S.append(value)
        else:
            self.S[self.SP + 1] = value
        self.SP += 1

    def stack_top(self):
        return self.S[self.SP]


class StackFrame(object):

    def __init__(self, vm, pre_stack_frame):
        self.vm = vm
        # save pre stack frame as linked list
        self.pre_stack_frame = pre_stack_frame
        self.start_frame_addr = len(self.vm.S)  # start addr for val find
        self.env = {}  # variable address dict

    def address(self, name):
        if self.env.get(name) is None:
            logging.warning(
                "find variable [%s] in current stack frame env error" % name)
        else:
            return self.env.get(name)

    # stack frame assign should already finished before interpret whole
    # program(static allocation)
    def assign(self, name, value):
        self.vm.stack_push(value)
        self.env[name] = self.vm.SP
        pass


if __name__ == '__main__':
    # test 1
    # (1+2)*3
    vm = VM()
    exp = BinaryExpression(vm, BinaryExpression(vm, 1, 2, '+'), 3, '*')
    exp.push_code()
    vm.run()
    logging.warning('result of (1+2)*3 is %d' % vm.S[-1])

    # test 2
    # x = 3 * 2
    vm = VM()
    vm.static_assign('x', None)
    AssignmentExpression(vm, 'x', BinaryExpression(vm, 3, 2, '*')).push_code()
    vm.run()
    logging.warn('result of x = 3*2 %d' % vm.inspect('x'))

    # test 3
    # x = 10, y = 5
    # if ((x+y) > 10) x = 1024; else y = 42;
    vm = VM()
    vm.static_assign('x', 10)
    vm.static_assign('y', 5)

    e1 = BinaryExpression(vm, BinaryExpression(vm, 'x', 'y', '+'), 10, '>')
    s1 = [Statement(AssignmentExpression(vm, 'x', 1024))]
    s2 = [Statement(AssignmentExpression(vm, 'y', 42))]

    if_statement = IfStatement(vm, e1, s1, s2)
    if_statement.push_code()

    vm.run()
    logging.warn('result of x = %d' % vm.inspect('x'))
    logging.warn('result of y = %d' % vm.inspect('y'))

    # test 4
    # x = 0 sum = 0 count 0
    # while (x <= 100) sum = sum + x; count = count + 1; x = x + 1;
    vm = VM()
    vm.static_assign('x', 0)
    vm.static_assign('sum', 0)
    vm.static_assign('count', 0)

    e1 = BinaryExpression(vm, 'x', 100, '<=')
    s1 = [
        Statement(AssignmentExpression(
            vm, 'sum', BinaryExpression(vm, 'sum', 'x', '+'))),
        Statement(AssignmentExpression(
            vm, 'count', BinaryExpression(vm, 'count', 1, '+'))),
        Statement(
            AssignmentExpression(vm, 'x', BinaryExpression(vm, 'x', 1, '+')))
    ]

    while_statement = WhileStatement(vm, e1, s1)
    while_statement.push_code()

    vm.run()
    logging.warn('result of sum = %d' % vm.inspect('sum'))
    logging.warn('result of count = %d' % vm.inspect('count'))

    # test 5
    # x = 0 sum = 0
    # for( x = 1 ; x <= 100; x = x + 1) sum = sum + x ; print sum

    vm = VM()
    vm.static_assign('x', 0)
    vm.static_assign('sum', 0)

    e1 = AssignmentExpression(vm, 'x', 1)
    e2 = BinaryExpression(vm, 'x', 100, '<=')
    e3 = AssignmentExpression(vm, 'x', BinaryExpression(vm, 'x', 1, '+'))

    s1 = [
        Statement(AssignmentExpression(
            vm, 'sum', BinaryExpression(vm, 'sum', 'x', '+'))),
        #PrintStatement('sum'),
    ]

    for_statement = ForStatement(vm, e1, e2, e3, s1)
    for_statement.push_code()
    vm.run()
    logging.warn('result of sum = %d' % vm.inspect('sum'))


    # test 6
    # x = 0 sum = 0
    # for( x = 1; x <= 100; x = x + 1 )
    #     while(sum < 10)
    #           sum + = 1
    #           print sum
    #           continue
    #     sum += x

    vm = VM()
    vm.static_assign('x', 0)
    vm.static_assign('sum', 0)
    vm.S = [0, 0, 1, 1, 1, 1, 1, 1]

    for_e1 = AssignmentExpression(vm, 'x', 1)
    for_e2 = BinaryExpression(vm, 'x', 100, '<=')
    for_e3 = AssignmentExpression(vm, 'x', BinaryExpression(vm, 'x', 1, '+'))

    while_e1 = BinaryExpression(vm, 'sum', 10, '<')
    while_s1 = [
        Statement(AssignmentExpression(
            vm, 'sum', BinaryExpression(vm, 'sum', 1, '+'))),
        PrintStatement('sum'),
        ContinueStatement()
    ]

    while_statement = WhileStatement(vm, while_e1, while_s1)

    add_s1 = Statement(AssignmentExpression(
            vm, 'sum', BinaryExpression(vm, 'sum', 'x', '+')))

    for_statement = ForStatement(vm, for_e1, for_e2, for_e3, [while_statement, add_s1])
    for_statement.push_code()
    vm.run()
    logging.warn('result of sum = %d' % vm.inspect('sum'))





