
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
    'E1': step.Buffer(step.Scalar("float"), [D]),
}
input_data = {
    'E0': torch.rand(N_value, M_value),
    'E1': torch.rand(N_value, M_value, D_value),
}

class MaxSum(step.Fn):
    def __init__(self, input, output):
        super().__init__("MaxSum", input, output)

    def getInit(self):
        return [torch.tensor(float('-inf')), torch.tensor(0), torch.zeros(D_value)]

    def apply(self, state, input):
        m_t, l_t, o_t = state # scalar, scalar, [D]
        s_t, v_t = input # scalar, [D]
        m_next = torch.max(m_t, s_t) # scalar
        l_prim_t = torch.exp(m_t - m_next) * l_t
        p_t = torch.exp(s_t - m_next)
        l_next = p_t + l_prim_t
        o_next = l_prim_t * o_t / l_next + p_t * v_t / l_next
        return [m_next, l_next, o_next]
        
    
fn_maxsum = MaxSum(step.STuple((step.Scalar("float"), step.Buffer(step.Scalar("float"), [D]))), step.STuple((step.Scalar("float"), step.Scalar("float"), step.Buffer(step.Scalar("float"), [D]))))
    

class GetThird(step.Fn):
    def __init__(self, input, output):
        super().__init__("GetThird", input, output)
    
    def apply(self, input):
        return [input[2]]
        
    
fn_getthird = GetThird(step.STuple((step.Scalar("float"), step.Scalar("float"), step.Buffer(step.Scalar("float"), [D]))), step.Buffer(step.Scalar("float"), [D]))
    

def prepare():
    E0 = step.Stream("E0", step.Scalar("float"), 1, [M, N])
    E0.ctx = ctx
    E0.data = [input_data['E0']]
    
    E1 = step.Stream("E1", step.Buffer(step.Scalar("float"), [D]), 1, [M, N])
    E1.ctx = ctx
    E1.data = [input_data['E1']]
    return E0, E1
def check_shape(S0):
    output_dtype_S0 = step.Scalar("float")
    assert S0.dtype == step.Scalar("float"), f"The output dtype should be {output_dtype_S0.dtype} but got {S0.dtype}"
    assert S0.shape == [D, N], f"The output shape should be [D, N] but got {S0.shape}"
    
def check_data(S0):
    
    S0_data_0 = torch.bmm(torch.softmax(input_data['E0'], 1).unsqueeze(1), input_data['E1']).squeeze(1)
    torch.testing.assert_close(S0.data[0], S0_data_0)
    

def test():
    E0, E1 = prepare()
    S0 = body(E0, E1)
    check_shape(S0)
    check_data(S0)
    
