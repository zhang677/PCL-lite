
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


E_value = 2
ctx[E] = E_value
gelu = torch.nn.GELU()


input_dtype = {
    'E0': step.Buffer(step.Scalar("float"), [K]),
    'E1': step.Scalar("float"),
    'E2': step.Scalar("float"),
}
input_data = {
    'E0': torch.randn(M_value, K_value),
    'E1': generate_binary_tensor((M_value)),
    'E2': torch.randn(M_value),
    'W': torch.randn(K_value, K_value),
}

class Expert0(step.Fn):
    def __init__(self, input, output):
        super().__init__("Expert0", input, output)
    
    def apply(self, input):
        return [gelu(input[0] @ input_data['W']) * input[1] + input[0]]
        
    
fn_expert0 = Expert0(step.STuple((step.Buffer(step.Scalar("float"), [K]), step.Scalar("float"))), step.Buffer(step.Scalar("float"), [K]))
    

class Expert1(step.Fn):
    def __init__(self, input, output):
        super().__init__("Expert1", input, output)
    
    def apply(self, input):
        return [input[0]]
        
    
fn_expert1 = Expert1(step.STuple((step.Buffer(step.Scalar("float"), [K]), step.Scalar("float"))), step.Buffer(step.Scalar("float"), [K]))
    

class Sum(step.Fn):
    def __init__(self, input, output):
        super().__init__("Sum", input, output)

    def getInit(self):
        return [torch.zeros(K_value)]

    def apply(self, state, input):
        return [input[0] + state[0]]
        
    
fn_sum = Sum(step.Buffer(step.Scalar("float"), [K]), step.Buffer(step.Scalar("float"), [K]))
    

class Filter(step.Fn):
    def __init__(self, input, output):
        super().__init__("Filter", input, output)
    
    def apply(self, input):
        return [torch.tensor([1.0, 0.0])] if (input[0] == 1.0) else [torch.tensor([0.0, 1.0])]
        
    
fn_filter = Filter(step.Scalar("float"), step.Multihot(step.Scalar("float"), E))
    

def prepare():
    E0 = step.Stream("E0", step.Buffer(step.Scalar("float"), [K]), 0, [M])
    E0.ctx = ctx
    E0.data = [input_data['E0']]
    
    E1 = step.Stream("E1", step.Scalar("float"), 0, [M])
    E1.ctx = ctx
    E1.data = [input_data['E1']]
    
    E2 = step.Stream("E2", step.Scalar("float"), 0, [M])
    E2.ctx = ctx
    E2.data = [input_data['E2']]
    return E0, E1, E2
def check_shape(S0):
    output_dtype_S0 = step.Buffer(step.Scalar("float"), [K])
    assert S0.dtype == step.Buffer(step.Scalar("float"), [K]), f"The output dtype should be {output_dtype_S0.dtype} but got {S0.dtype}"
    assert S0.shape == [M], f"The output shape should be [M] but got {S0.shape}"
    
def check_data(S0):
    E0_data = input_data['E0']
    score = input_data['E1'].unsqueeze(-1) 
    affinity = input_data['E2'].unsqueeze(-1)
    S0_data_0 = (affinity * (gelu(E0_data @ input_data['W'])) + E0_data) * score + E0_data * (1 - score)
    torch.testing.assert_close(S0.data[0], S0_data_0)
    

def test():
    E0, E1, E2 = prepare()
    S0 = body(E0, E1, E2)
    check_shape(S0)
    check_data(S0)
    
