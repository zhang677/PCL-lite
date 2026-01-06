
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
sigmoid = torch.nn.Sigmoid()
gelu = torch.nn.GELU()


input_dtype = {
    'E0': step.Scalar("float"),
}
input_data = {
    'E0': torch.randn(M_value, K_value),
    'Wg': torch.randn(K_value),
    'W': torch.randn(K_value, K_value),
}

class Gate(step.Fn):
    def __init__(self, input, output):
        super().__init__("Gate", input, output)
    
    def apply(self, input):
        return [sigmoid(input[0] @ input_data['Wg'])]
        
    
fn_gate = Gate(step.Buffer(step.Scalar("float"), [K]), step.Scalar("float"))
    

class Top3(step.Fn):
    def __init__(self, input, output):
        super().__init__("Top3", input, output)
    
    def apply(self, input):
        _, indices = torch.topk(input[0], 3, dim=0)
        multihot = torch.zeros_like(input[0], dtype=torch.float32)
        multihot.scatter_(0, indices, 1.0)
        return [multihot]
        
    
fn_top3 = Top3(step.Buffer(step.Scalar("float"), [M]), step.Buffer(step.Scalar("float"), [M]))
    

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
    E0 = step.Stream("E0", step.Scalar("float"), 1, [K, M])
    E0.ctx = ctx
    E0.data = [input_data['E0']]
    return E0
def check_shape(S0):
    output_dtype_S0 = step.Buffer(step.Scalar("float"), [K])
    assert S0.dtype == step.Buffer(step.Scalar("float"), [K]), f"The output dtype should be {output_dtype_S0.dtype} but got {S0.dtype}"
    assert S0.shape == [M], f"The output shape should be [M] but got {S0.shape}"
    
def check_data(S0):
    E0_data = input_data['E0']
    affinity = sigmoid(E0_data @ input_data['Wg']) # [M]
    _, indices = torch.topk(affinity, 3, dim=0)
    multihot = torch.zeros_like(affinity, dtype=torch.float32)
    score = multihot.scatter(0, indices, 1.0).unsqueeze(-1) # [M]
    S0_data_0 = (affinity.unsqueeze(-1) * (gelu(E0_data @ input_data['W'])) + E0_data) * score + E0_data * (1 - score)
    torch.testing.assert_close(S0.data[0], S0_data_0)
    

def test():
    E0 = prepare()
    S0 = body(E0)
    check_shape(S0)
    check_data(S0)
    
