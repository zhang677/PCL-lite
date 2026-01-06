
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
softmax = torch.nn.Softmax(dim=-1)
gelu = torch.nn.GELU()


input_dtype = {
    'E0': step.Scalar("float"),
}
input_data = {
    'E0': torch.randn(M_value, N_value, K_value),
    'Wg': torch.randn(K_value, E_value),
    'W0_0': torch.randn(K_value, D_value),
    'W0_1': torch.randn(D_value, K_value),
    'W1_0': torch.randn(K_value, D_value),
    'W1_1': torch.randn(D_value, K_value),
    'W2_0': torch.randn(K_value, D_value),
    'W2_1': torch.randn(D_value, K_value),
}

class Gate(step.Fn):
    def __init__(self, input, output):
        super().__init__("Gate", input, output)
    
    def apply(self, input):
        return [softmax(input[0] @ input_data['Wg'])]
        
    
fn_gate = Gate(step.Buffer(step.Scalar("float"), [K]), step.Buffer(step.Scalar("float"), [E]))
    

class Top2(step.Fn):
    def __init__(self, input, output):
        super().__init__("Top2", input, output)
    
    def apply(self, input):
        _, indices = torch.topk(input[0], 2, dim=-1)
        multihot = torch.zeros_like(input[0], dtype=torch.float32)
        multihot.scatter_(-1, indices, 1.0)
        return [multihot]
        
    
fn_top2 = Top2(step.Buffer(step.Scalar("float"), [E]), step.Multihot(step.Scalar("float"), E))
    

class Sum(step.Fn):
    def __init__(self, input, output):
        super().__init__("Sum", input, output)

    def getInit(self):
        return [torch.zeros(K_value)]

    def apply(self, state, input):
        return [state[0] + input[0]]
        
    
fn_sum = Sum(step.Buffer(step.Scalar("float"), [K]), step.Buffer(step.Scalar("float"), [K]))
    

class Expert0(step.Fn):
    def __init__(self, input, output):
        super().__init__("Expert0", input, output)
    
    def apply(self, input):
        return [gelu(input[0] @ input_data['W0_0']) @ input_data['W0_1'] * input[1][0]]
        
    
fn_expert0 = Expert0(step.STuple((step.Buffer(step.Scalar("float"), [K]), step.Buffer(step.Scalar("float"), [E]))), step.Buffer(step.Scalar("float"), [K]))
    

class Expert1(step.Fn):
    def __init__(self, input, output):
        super().__init__("Expert1", input, output)
    
    def apply(self, input):
        return [gelu(input[0] @ input_data['W1_0']) @ input_data['W1_1'] * input[1][1]]
        
    
fn_expert1 = Expert1(step.STuple((step.Buffer(step.Scalar("float"), [K]), step.Buffer(step.Scalar("float"), [E]))), step.Buffer(step.Scalar("float"), [K]))
    

class Expert2(step.Fn):
    def __init__(self, input, output):
        super().__init__("Expert2", input, output)
    
    def apply(self, input):
        return [gelu(input[0] @ input_data['W2_0']) @ input_data['W2_1'] * input[1][2]]
        
    
fn_expert2 = Expert2(step.STuple((step.Buffer(step.Scalar("float"), [K]), step.Buffer(step.Scalar("float"), [E]))), step.Buffer(step.Scalar("float"), [K]))
    

def prepare():
    E0 = step.Stream("E0", step.Scalar("float"), 2, [K, N, M])
    E0.ctx = ctx
    E0.data = [input_data['E0']]
    return E0
def check_shape(S0):
    output_dtype_S0 = step.Buffer(step.Scalar("float"), [K])
    assert S0.dtype == step.Buffer(step.Scalar("float"), [K]), f"The output dtype should be {output_dtype_S0.dtype} but got {S0.dtype}"
    assert S0.shape == [N, M], f"The output shape should be [N, M] but got {S0.shape}"
    
def check_data(S0):
    affinity = softmax(input_data['E0'] @ input_data['Wg'])
    _, indices = torch.topk(affinity, 2, dim=-1)
    multihot = torch.zeros_like(affinity, dtype=torch.float32)
    index_tensor = multihot.scatter(-1, indices, 1.0).movedim(-1, 0)
    W0 = torch.stack([input_data[f'W{i}_0'] for i in range(E_value)])
    W1 = torch.stack([input_data[f'W{i}_1'] for i in range(E_value)])
    hidden = torch.einsum('mnk,ekh->emnh', input_data['E0'], W0)
    hidden = gelu(hidden)
    transformed = torch.einsum('emnh,ehk->emnk', hidden, W1)
    weighted = transformed * affinity.movedim(-1, 0).unsqueeze(-1)
    S0_data_0 = (index_tensor.unsqueeze(-1) * weighted).sum(dim=0)
    torch.testing.assert_close(S0.data[0], S0_data_0)
    

def test():
    E0 = prepare()
    S0 = body(E0)
    check_shape(S0)
    check_data(S0)
    
