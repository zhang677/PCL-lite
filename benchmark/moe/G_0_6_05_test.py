
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


E_value = 3
ctx[E] = E_value
gelu = torch.nn.GELU()


input_dtype = {
    'E0': step.Buffer(step.Scalar("float"), [K]),
    'E1': step.Multihot(step.Scalar("float"), E),
    'E2': step.Buffer(step.Scalar("float"), [E]),
}
input_data = {
    'E0': torch.randn(M_value, N_value, K_value),
    'E1': generate_multi_hot((M_value, N_value), 0, 3, E_value),
    'E2': torch.randn(M_value, N_value, E_value),
    'W0_0': torch.randn(K_value, D_value),
    'W0_1': torch.randn(D_value, K_value),
    'W1_0': torch.randn(K_value, D_value),
    'W1_1': torch.randn(D_value, K_value),
    'W2_0': torch.randn(K_value, D_value),
    'W2_1': torch.randn(D_value, K_value),
}

class WeightedSum(step.Fn):
    def __init__(self, input, output):
        super().__init__("WeightedSum", input, output)

    def getInit(self):
        return [torch.zeros(K_value)]

    def apply(self, state, input):
        return [state[0] + input[0] * input[1]]
        
    
fn_weighted_sum = WeightedSum(step.STuple((step.Buffer(step.Scalar("float"), [K]), step.Scalar("float"))), step.Buffer(step.Scalar("float"), [K]))
    

class Expert0(step.Fn):
    def __init__(self, input, output):
        super().__init__("Expert0", input, output)
    
    def apply(self, input):
        return [gelu(input[0] @ input_data['W0_0']) @ input_data['W0_1'], input[1][0]]
        
    
fn_expert0 = Expert0(step.STuple((step.Buffer(step.Scalar("float"), [K]), step.Buffer(step.Scalar("float"), [E]))), step.STuple((step.Buffer(step.Scalar("float"), [K]), step.Scalar("float"))))
    

class Expert1(step.Fn):
    def __init__(self, input, output):
        super().__init__("Expert1", input, output)
    
    def apply(self, input):
        return [gelu(input[0] @ input_data['W1_0']) @ input_data['W1_1'], input[1][1]]
        
    
fn_expert1 = Expert1(step.STuple((step.Buffer(step.Scalar("float"), [K]), step.Buffer(step.Scalar("float"), [E]))), step.STuple((step.Buffer(step.Scalar("float"), [K]), step.Scalar("float"))))
    

class Expert2(step.Fn):
    def __init__(self, input, output):
        super().__init__("Expert2", input, output)
    
    def apply(self, input):
        return [gelu(input[0] @ input_data['W2_0']) @ input_data['W2_1'], input[1][2]]
        
    
fn_expert2 = Expert2(step.STuple((step.Buffer(step.Scalar("float"), [K]), step.Buffer(step.Scalar("float"), [E]))), step.STuple((step.Buffer(step.Scalar("float"), [K]), step.Scalar("float"))))
    

def prepare():
    E0 = step.Stream("E0", step.Buffer(step.Scalar("float"), [K]), 1, [N, M])
    E0.ctx = ctx
    E0.data = [input_data['E0']]
    
    E1 = step.Stream("E1", step.Multihot(step.Scalar("float"), E), 1, [N, M])
    E1.ctx = ctx
    E1.data = [input_data['E1']]
    
    E2 = step.Stream("E2", step.Buffer(step.Scalar("float"), [E]), 1, [N, M])
    E2.ctx = ctx
    E2.data = [input_data['E2']]
    return E0, E1, E2
def check_shape(S0):
    output_dtype_S0 = step.Buffer(step.Scalar("float"), [K])
    assert S0.dtype == step.Buffer(step.Scalar("float"), [K]), f"The output dtype should be {output_dtype_S0.dtype} but got {S0.dtype}"
    assert S0.shape == [N, M], f"The output shape should be [N, M] but got {S0.shape}"
    
def check_data(S0):
    index_tensor = input_data['E1'].movedim(-1, 0)
    E0 = input_data['E0']
    E2 = input_data['E2']
    W0 = torch.stack([input_data[f'W{i}_0'] for i in range(E_value)])
    W1 = torch.stack([input_data[f'W{i}_1'] for i in range(E_value)])
    hidden = torch.einsum('mnk,ekh->emnh', E0, W0)
    hidden = gelu(hidden)
    transformed = torch.einsum('emnh,ehk->emnk', hidden, W1)
    weighted = transformed * E2.movedim(-1, 0).unsqueeze(-1)
    S0_data_0 = (index_tensor.unsqueeze(-1) * weighted).sum(dim=0)
    torch.testing.assert_close(S0.data[0], S0_data_0)
    

def test():
    E0, E1, E2 = prepare()
    S0 = body(E0, E1, E2)
    check_shape(S0)
    check_data(S0)
    
