__author__ = 'hzyuxin'

import operator
from functools import partial


class Expression(object):
    def __init__(self):
        pass

    def codegen(self):
        pass


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

    def __init__(self, left, right, op):
        super(BinaryExpression, self).__init__()
        self.left = left
        self.right = right
        self.op = op

    def codegen(self):

        if isinstance(self.left, Expression):
            codegen_l = self.left.codegen()
        else:
            codegen_l = ['loadc ' + str(self.left)]

        if isinstance(self.right, Expression):
            codegen_r = self.right.codegen()
        else:
            codegen_r = ['loadc ' + str(self.right)]

        return codegen_l + codegen_r + [BinaryExpression.op_dict[self.op]]


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
        self.vm.S.append(q)

    def load(self):
        address = self.vm.S.pop()
        self.vm.S.append(self.vm.S[address])

    def loada(self, address):
        self.vm.S.append(self.vm.S[address])

    def store(self):
        address = self.vm.S.pop()
        value = self.vm.S.top()
        self.vm.S[address] = value

    def storea(self, address):
        value = self.vm.S.top()
        self.vm.S[address] = value

    def jump(self, address):
        self.vm.PC = address

    def jumpz(self, address):
        value = self.vm.S.pop()
        if value is 0:
            self.vm.PC = address

    def pop(self):
        self.vm.S.pop()

    def binary_op(self, op_name):
        r = self.vm.S.pop()
        l = self.vm.S.pop()
        lam = self.binary_ops[op_name]
        self.vm.S.append(lam(l, r))

    def unary_op(self, op_name):
        l = self.vm.S.pop()
        lam = self.unary_ops[op_name]
        self.vm.S.append(lam(l))

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
        ir_args = [int(i) for i in ir_exp[1:]]
        getattr(self, ir_name)(*ir_args)


class VM(object):
    def __init__(self):
        self.S = []  # main memory
        self.C = []  # program ir stack
        self.PC = 0  # program counter

        self.IR = IR(self)

    def run(self):
        while self.PC != len(self.C):
            ir_exp = self.C[self.PC]
            self.IR.execute_ir(ir_exp)
            self.PC += 1

if __name__ == '__main__':
    exp = BinaryExpression(BinaryExpression(1, 2, '+'), 3, '*')
    vm = VM()
    vm.C = exp.codegen()
    vm.run()

    pass