
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
    'E0': step.Buffer(step.Scalar("float"), [K]),
}
input_data = {
    'E0': torch.rand(D_value, K_value),
}

def prepare():
    E0 = step.Stream("E0", step.Buffer(step.Scalar("float"), [K]), 0, [D])
    E0.ctx = ctx
    E0.data = [input_data['E0']]
    return E0
def check_shape(S0, S1):
    assert S0.shape == [D]
    assert S0.dtype == step.Buffer(step.Scalar("float"), [K])
    
    assert S1.shape == [D]
    assert S1.dtype == step.Buffer(step.Scalar("float"), [K])
    
def check_data(S0, S1):
    S0_data_0 = input_data['E0']
 
    torch.testing.assert_close(S0.data[0], S0_data_0)
    
    S1_data_0 = input_data['E0']
 
    torch.testing.assert_close(S1.data[0], S1_data_0)
    

def test():
    E0 = prepare()
    S0, S1 = body(E0)
    check_shape(S0, S1)
    check_data(S0, S1)
    

def body(E0):
    E1, E2 = step.Copy().apply(E0)
    return E1, E2
    
