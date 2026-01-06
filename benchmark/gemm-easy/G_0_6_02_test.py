
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
    'E0': torch.rand(N_value, M_value, K_value),
    'E1': torch.rand(N_value, K_value),
}

class Mul(step.Fn):
    def __init__(self, input, output):
        super().__init__("Mul", input, output)
    
    def apply(self, input):
        return [input[0] * input[1]]
        
    
fn_mul = Mul(step.STuple((step.Scalar("float"), step.Scalar("float"))), step.Scalar("float"))
    

class Add(step.Fn):
    def __init__(self, input, output):
        super().__init__("Add", input, output)

    def getInit(self):
        return [torch.tensor(0)]

    def apply(self, state, input):
        return [state[0] + input[0]]
        
    
fn_add = Add(step.Scalar("float"), step.Scalar("float"))
    

def prepare():
    E0 = step.Stream("E0", step.Scalar("float"), 2, [K, M, N])
    E0.ctx = ctx
    E0.data = [input_data['E0']]
    
    E1 = step.Stream("E1", step.Scalar("float"), 1, [K, N])
    E1.ctx = ctx
    E1.data = [input_data['E1']]
    return E0, E1
def check_shape(S0):
    output_dtype_S0 = step.Scalar("float")
    assert S0.dtype == step.Scalar("float"), f"The output dtype should be {output_dtype_S0.dtype} but got {S0.dtype}"
    assert S0.shape == [M, N], f"The output shape should be [M, N] but got {S0.shape}"
    
def check_data(S0):
    
    S0_data_0 = (input_data['E0'] * input_data['E1'].unsqueeze(1).repeat(1, M_value, 1)).sum(2)
    torch.testing.assert_close(S0.data[0], S0_data_0)
    

def test():
    E0, E1 = prepare()
    S0 = body(E0, E1)
    check_shape(S0)
    check_data(S0)
    
