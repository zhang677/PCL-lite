
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
    'E1': step.Buffer(step.Scalar("float"), [D]),
}
input_data = {
    'E0': torch.rand(N_value, M_value),
    'E1': torch.rand(N_value, M_value, D_value),
}

def prepare():
    E0 = step.Stream("E0", step.Scalar("float"), 1, [M, N])
    E0.ctx = ctx
    E0.data = [input_data['E0']]
    
    E1 = step.Stream("E1", step.Buffer(step.Scalar("float"), [D]), 1, [M, N])
    E1.ctx = ctx
    E1.data = [input_data['E1']]
    return E0, E1
def check_shape(S0):
    assert S0.shape == [M, N]
    assert S0.dtype == step.STuple((step.Scalar("float"), step.Buffer(step.Scalar("float"), [D])))
    
def check_data(S0):
    S0_data_0 = input_data['E0']
 
    torch.testing.assert_close(S0.data[0], S0_data_0)
    
    S0_data_1 = input_data['E1']
 
    torch.testing.assert_close(S0.data[1], S0_data_1)
    

def test():
    E0, E1 = prepare()
    S0 = body(E0, E1)
    check_shape(S0)
    check_data(S0)
    

def body(E0, E1):
    E2 = step.Zip().apply((E0, E1))
    return E2
    
