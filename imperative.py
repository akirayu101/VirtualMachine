__author__ = 'hzyuxin'

import operator
from functools import partial


class Code(object):
    pass


class IR(object):
    def __init__(self, vm):
        self.vm = vm

        self.binary_ops = {
            'add': operator.add,
            'sub': operator.sub,
            'div': operator.div,
            'mod': operator.mod,
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

    def load_const(self, q):
        self.vm.S.append(q)

    def loadc(self):
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
    imperative_vm = VM()

    imperative_vm.S.append(1)
    imperative_vm.S.append(4)

    imperative_vm.C.append('sub')

    imperative_vm.run()

    pass