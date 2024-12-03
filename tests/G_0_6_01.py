
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

class ExpMaxDiff(step.Fn):
    def __init__(self, input, output):
        super().__init__("ExpMaxDiff", input, output)

    def getInit(self):
        return [torch.tensor(float('-inf')), torch.tensor(0), torch.tensor(0)]

    def apply(self, state, input):
        m_t, e_t, d_t = state # scalar, scalar, scalar
        s_t = input[0]
        m_next = torch.max(m_t, s_t)
        dm_next = m_t - m_next
        e_next = torch.exp(s_t - m_next)
        d_next = torch.exp(dm_next)
        return [m_next, e_next, d_next]
        
    
fn_expmaxdiff = ExpMaxDiff(step.Scalar("float"), step.STuple((step.Scalar("float"), step.Scalar("float"), step.Scalar("float"))))
    

class DivSum(step.Fn):
    def __init__(self, input, output):
        super().__init__("DivSum", input, output)

    def getInit(self):
        return [torch.zeros(D_value), torch.zeros(D_value)]

    def apply(self, state, input):
        v_t, e_t, d_t = input # [D], scalar, scalar
        l_t, o_t = state # sclar, [D]
        l_prim_t = d_t * l_t
        l_next = e_t + l_prim_t
        o_next = l_prim_t * o_t / l_next + e_t * v_t / l_next
        return [l_next, o_next]
        
    
fn_divsum = DivSum(step.STuple((step.Buffer(step.Scalar("float"), [D]), step.Scalar("float"), step.Scalar("float"))), step.STuple((step.Buffer(step.Scalar("float"), [D]), step.Buffer(step.Scalar("float"), [D]))))
    

class GetSecondThird(step.Fn):
    def __init__(self, input, output):
        super().__init__("GetSecondThird", input, output)
    
    def apply(self, input):
        return [input[1], input[2]]
        
    
fn_getsecondthird = GetSecondThird(step.STuple((step.Scalar("float"), step.Scalar("float"), step.Scalar("float"))), step.STuple((step.Scalar("float"), step.Scalar("float"))))
    

class GetSecond(step.Fn):
    def __init__(self, input, output):
        super().__init__("GetSecond", input, output)
    
    def apply(self, input):
        return [input[1]]
        
    
fn_getsecond = GetSecond(step.STuple((step.Buffer(step.Scalar("float"), [D]), step.Buffer(step.Scalar("float"), [D]))), step.Buffer(step.Scalar("float"), [D]))
    

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
    
def body(E0, E1):
    pass