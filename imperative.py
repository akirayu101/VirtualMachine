__author__ = 'hzyuxin'

import operator
import logging
from functools import partial


class Expression(object):
    def __init__(self, vm):
        self.vm = vm

    def codegen(self):
        pass

    @staticmethod
    def codegen_expression(e):
        if isinstance(e, Expression):
            return e.codegen()
        else:
            return ['loadc ' + str(e)]


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

        codegen_l = Expression.codegen_expression(self.left)
        codegen_r = Expression.codegen_expression(self.right)

        return codegen_l + codegen_r + [BinaryExpression.op_dict[self.op]]


class AssignmentExpression(Expression):

    def __init__(self, vm, left, right):
        super(AssignmentExpression, self).__init__(vm)
        self.left = left
        self.right = right

    def codegen(self):
        address = self.vm.address(self.left)

        codegen_l = ['loadc ' + str(address)]
        codegen_r = Expression.codegen_expression(self.right)
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
        self.vm.S.append(q)

    def load(self):
        address = self.vm.S.pop()
        self.vm.S.append(self.vm.S[address])

    def loada(self, address):
        self.vm.S.append(self.vm.S[address])

    def store(self):
        address = self.vm.S.pop()
        value = self.vm.S[-1]
        self.vm.S[address] = value

    def storea(self, address):
        value = self.vm.S[-1]
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
        self.stack_frame = StackFrame(self, None)

    def run(self):
        while self.PC != len(self.C):
            ir_exp = self.C[self.PC]
            self.IR.execute_ir(ir_exp)
            self.PC += 1

    def address(self, name):
        return self.stack_frame.address(name)

    def static_assign(self, name, value):
        self.stack_frame.assign(name, value)

    def inspect(self, name):
        address = self.address(name)
        return self.S[address]


class StackFrame(object):
    def __init__(self, vm, pre_stack_frame):
        self.vm = vm
        self.pre_stack_frame = pre_stack_frame  # save pre stack frame as linked list
        self.start_frame_addr = len(self.vm.S)  # start addr for val find
        self.env = {}                           # variable address dict

    def address(self, name):
        if self.env.get(name) is None:
            logging.warning("find variable [%s] in current stack frame env error" % name)
        else:
            return self.env.get(name)

    # stack frame assign should already finished before interpret whole program(static allocation)
    def assign(self, name, value):
        self.vm.S.append(value)
        self.env[name] = len(self.vm.S) - self.start_frame_addr - 1
        pass


if __name__ == '__main__':

    # test 1 (1+2)*3
    vm = VM()
    exp = BinaryExpression(vm, BinaryExpression(vm, 1, 2, '+'), 3, '*')
    vm.C = exp.codegen()
    vm.run()
    logging.warning('result of (1+2)*3 is %d' % vm.S[-1])


    # test x = 1
    vm = VM()
    vm.static_assign('x', None)
    vm.C = AssignmentExpression(vm, 'x', BinaryExpression(vm, 3, 2, '*')).codegen()
    vm.run()
    logging.warn('result of x = 3*2 %d' % vm.inspect('x'))