
import step
from sympy import Symbol
import torch
from tools.get_indices import generate_multi_hot, generate_binary_tensor

torch.manual_seed(42)
E = Symbol("E")
M = Symbol("M")
N = Symbol("N")
K = Symbol("K")
D = Symbol("D")
M_value = 5
N_value = 7
K_value = 9
D_value = 16
ctx = {
    M: M_value,
    N: N_value,
    K: K_value,
    D: D_value
}

input_dtype = {
    'E0': step.Scalar("float"),
}
input_data = {
    'E0': torch.rand(M_value, N_value, D_value),
}

class Square(step.Fn):
    def __init__(self, input, output):
        super().__init__("Square", input, output)
    
    def apply(self, input):
        return [input[0] * input[0]]
        
    
fn_square = Square(step.Scalar("float"), step.Scalar("float"))
    

class AccumAdd(step.Fn):
    def __init__(self, input, output):
        super().__init__("AccumAdd", input, output)

    def getInit(self):
        return [torch.tensor(0)]

    def apply(self, state, input):
        return [state[0] + input[0]]
        
    
fn_accumadd = AccumAdd(step.Scalar("float"), step.Scalar("float"))
    

class MulByD(step.Fn):
    def __init__(self, input, output):
        super().__init__("MulByD", input, output)
    
    def apply(self, input):
        return [input[0] * D_value]
        
    
fn_mulbyD = MulByD(step.Scalar("float"), step.Scalar("float"))
    

class PowByD(step.Fn):
    def __init__(self, input, output):
        super().__init__("PowByD", input, output)
    
    def apply(self, input):
        return [torch.pow(input[0], D_value)]
        
    
fn_powbyD = PowByD(step.Scalar("float"), step.Scalar("float"))
    

class AddbyD(step.Fn):
    def __init__(self, input, output):
        super().__init__("AddbyD", input, output)
    
    def apply(self, input):
        return [input[0] + D_value]
        
    
fn_addbyD = AddbyD(step.Scalar("float"), step.Scalar("float"))
    

class Div(step.Fn):
    def __init__(self, input, output):
        super().__init__("Div", input, output)
    
    def apply(self, input):
        return [input[0] / input[1]]
        
    
fn_div = Div(step.STuple((step.Scalar("float"), step.Scalar("float"))), step.Scalar("float"))
    

class Mul(step.Fn):
    def __init__(self, input, output):
        super().__init__("Mul", input, output)
    
    def apply(self, input):
        return [input[0] * input[1]]
        
    
fn_mul = Mul(step.STuple((step.Scalar("float"), step.Scalar("float"))), step.Scalar("float"))
    

class Sub(step.Fn):
    def __init__(self, input, output):
        super().__init__("Sub", input, output)
    
    def apply(self, input):
        return [input[0] - input[1]]
        
    
fn_sub = Sub(step.STuple((step.Scalar("float"), step.Scalar("float"))), step.Scalar("float"))
    

def test(): 

    
    S0_data_0 = torch.ops.aten.native_layer_norm.default(input_data['E0'], [D_value], None, None, 1e-5)[0] 
    assert S0_data_0.shape == (M_value, N_value, D_value)
    