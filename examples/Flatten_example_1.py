
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
    'E0': torch.rand(K_value, N_value, M_value),
}

def prepare():
    E0 = step.Stream("E0", step.Scalar("float"), 2, [M, N, K])
    E0.ctx = ctx
    E0.data = [input_data['E0']]
    return E0
def check_shape(S0):
    assert S0.shape == [M, N*K]
    assert S0.dtype == step.Scalar("float")
    
def check_data(S0):
    S0_data_0 = input_data['E0'].view(-1, input_data['E0'].size(2))
 
    torch.testing.assert_close(S0.data[0], S0_data_0)
    

def test():
    E0 = prepare()
    S0 = body(E0)
    check_shape(S0)
    check_data(S0)
    

def body(E0):
    E1 = step.Flatten(L=[2,]).apply(E0)
    return E1
