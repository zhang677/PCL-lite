
import step
from sympy import Symbol
import torch

M = Symbol("M")
N = Symbol("N")
K = Symbol("K")
D = Symbol("D")
M_value = 5
N_value = 7
K_value = 9
D_value = 11
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
    'E0': torch.rand(N_value, M_value),
}

class Sum(step.Fn):
    def __init__(self, input, output):
        super().__init__("Sum", input, output)

    def getInit(self):
        return [torch.tensor(0)]

    def apply(self, state, input):
        return [state[0] + input[0]]
        
    
fn_sum = Sum(step.Scalar("float"), step.Scalar("float"))
    

def prepare():
    E0 = step.Stream("E0", step.Scalar("float"), 1, [M, N])
    E0.ctx = ctx
    E0.data = [input_data['E0']]
    return E0
def check_shape(S0):
    assert S0.shape == [N]
    assert S0.dtype == step.Scalar("float")
    
def check_data(S0):
    S0_data_0 = torch.sum(input_data['E0'], 1, keepdim=False)
 
    torch.testing.assert_close(S0.data[0], S0_data_0)
    

def test():
    E0 = prepare()
    S0 = body(E0)
    check_shape(S0)
    check_data(S0)
    

def body(E0):
    E1 = step.Accum(fn=fn_sum, b=1).apply(E0)
    return E1
    
