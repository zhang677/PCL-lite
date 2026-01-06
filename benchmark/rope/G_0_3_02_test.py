
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
    'E0': torch.ones((M_value, N_value), dtype=torch.float),
    'E1': torch.randn(M_value, N_value, D_value),
}

class Sum(step.Fn):
    def __init__(self, input, output):
        super().__init__("Sum", input, output)

    def getInit(self):
        return [torch.tensor(-1)]

    def apply(self, state, input):
        return [input[0] + state[0]]
        
    
fn_sum = Sum(step.Scalar("float"), step.Scalar("float"))
    

class NeoXRoPE(step.Fn):
    def __init__(self, input, output):
        super().__init__("NeoXRoPE", input, output)
    
    def apply(self, input):
        freq = input[0] / (10000**(torch.arange(0, D_value, 2, dtype=torch.float) / ctx[D]))
        cos = freq.cos()
        sin = freq.sin()
        x1, x2 = input[1].chunk(2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        return [torch.cat([o1, o2], dim=-1)]
        
    
fn_neoxrope = NeoXRoPE(step.STuple((step.Scalar("float"), step.Buffer(step.Scalar("float"), [D]))), step.Buffer(step.Scalar("float"), [D]))
    

def prepare():
    E0 = step.Stream("E0", step.Scalar("float"), 1, [N, M])
    E0.ctx = ctx
    E0.data = [input_data['E0']]
    
    E1 = step.Stream("E1", step.Buffer(step.Scalar("float"), [D]), 1, [N, M])
    E1.ctx = ctx
    E1.data = [input_data['E1']]
    return E0, E1
def check_shape(S0):
    output_dtype_S0 = step.Buffer(step.Scalar("float"), [D])
    assert S0.dtype == step.Buffer(step.Scalar("float"), [D]), f"The output dtype should be {output_dtype_S0.dtype} but got {S0.dtype}"
    assert S0.shape == [N, M], f"The output shape should be [N, M] but got {S0.shape}"
    
def check_data(S0):
    inv_freq = 1.0 / (10000**(torch.arange(0, D_value, 2, dtype=torch.float) / D_value))
    t = torch.arange(N_value, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    x1, x2 = input_data['E1'].chunk(2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    S0_data_0 = torch.cat((o1, o2), dim=-1)
    torch.testing.assert_close(S0.data[0], S0_data_0)
    

def test():
    E0, E1 = prepare()
    S0 = body(E0, E1)
    check_shape(S0)
    check_data(S0)
    
