import sympy
from sympy import symbols
from sympy.logic import simplify_logic
from sympy.logic.boolalg import truth_table, Not, Or, And, Xor, Xnor, Nand, Nor

def demo1():
    A, B, C = symbols('A B C')
    expression: Or = (A & B) | (A & ~B)
    print(expression)
    simplified_expression = simplify_logic(expression)
    print(simplified_expression)
    for a in truth_table(simplified_expression, [A]):
        print(a[0], a[1], type(a[1]))
    print(type(expression))
    print(expression.args[0])
    print(expression.args[1])

def demo2_rippleadder():
    def full_adder(a, b, cin):
        sum_ = Xor(a, b, cin)
        carry = Or(And(a, b), And(b, cin), And(a, cin))
        return sum_, carry
    A = symbols('A0 A1 A2 A3')
    B = symbols('B0 B1 B2 B3')
    Cin = symbols('Cin')
    S = []  # Sum bits
    carry = Cin  # Initial carry-in
    for i in range(3):
        sum_, carry = full_adder(A[i], B[i], carry)
        S.append(sum_)
    Cout = carry
    print(S)
    print(Cout)
    print(simplify_logic(Cout))
    print(simplify_logic(Cout).subs(Cin, 0))
    print(simplify_logic(S[0]).subs(Cin, 0))
    # print(simplify_logic(S[1]).subs(Cin, 0))
    # print(simplify_logic(S[2]).subs(Cin, 0))
    return S, Cout

if __name__ == "__main__":
    demo2_rippleadder