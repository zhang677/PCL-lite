
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
    'E1': step.Scalar("float"),
}
input_data = {
    'E0': torch.randn(M_value, N_value, K_value),
    'E1': torch.randn(M_value, K_value, D_value),
}

class Mul(step.Fn):
    def __init__(self, input, output):
        super().__init__("Mul", input, output)
    
    def apply(self, input):
        return [input[0] * input[1]]
        
    
fn_mul = Mul(step.STuple((step.Scalar("float"), step.Buffer(step.Scalar("float"), [D]))), step.Buffer(step.Scalar("float"), [D]))
    

class Sum(step.Fn):
    def __init__(self, input, output):
        super().__init__("Sum", input, output)

    def getInit(self):
        return [torch.zeros(D_value)]

    def apply(self, state, input):
        return [state[0] + input[0]]
        
    
fn_sum = Sum(step.Buffer(step.Scalar("float"), [D]), step.Buffer(step.Scalar("float"), [D]))
    

def test(): 

    
    S0_data_0 = torch.einsum('mnk,mkd->mnd', input_data['E0'], input_data['E1']) 
    assert S0_data_0.shape == (M_value, N_value, D_value)
    