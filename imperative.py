__author__ = 'hzyuxin'

import operator


class Code(object):
    pass


class IR(object):
    def __init__(self, vm):
        self.vm = vm
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

    def binary_op(self, lam):
        l = self.vm.S.pop()
        r = self.vm.S.pop()
        self.vm.S.append(lam(l, r))

    def unary_op(self, lam):
        l = self.vm.S.pop()
        self.vm.S.append(lam(l))

    def init_binary_ops(self):
        binary_ops = {
            'add': operator.add,
            'sub': operator.sub,
            'div': operator.div,
            'mod': operator.mod,
            'and': operator.iand,
            'or' : operator.ior,
            'eq' : operator.eq,
            'neq': lambda a, b: a != b,
            'le' : operator.lt,
            'leq': operator.le,
            'gr' : operator.gt,
            'geq': operator.ge,
        }

        for k, v in binary_ops.iteritems():
            setattr(self, k, lambda : self.binary_op(v))

    def init_unary_ops(self):
        unary_ops = {
            'neg': operator.neg,
            'not': operator.not_,
        }

        for k, v in unary_ops.iteritems():
            setattr(self, k, lambda : self.unary_op(v))


class VM(object):
    def __init__(self):
        self.S = []  # main memory
        self.C = []  # program ir stack
        self.PC = 0  # program counter

if __name__ == '__main__':
    vm = VM()
    ir = IR(vm)