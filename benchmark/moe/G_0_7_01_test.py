
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
    'E1': step.Multihot(step.Scalar("float"), E),
}
input_data = {
    'E0': torch.randn(M_value, N_value, D_value),
    'E1': generate_multi_hot((M_value, N_value), 1, 1, E_value),
    'W': torch.randn(D_value, D_value),
    'Wg': torch.randn(D_value),
}

class Expert0(step.Fn):
    def __init__(self, input, output):
        super().__init__("Expert0", input, output)
    
    def apply(self, input):
        return [input[1] * gelu(input[0] @ input_data['W']) + input[0]]
        
    
fn_expert0 = Expert0(step.STuple((step.Buffer(step.Scalar("float"), [D]), step.Scalar("float"))), step.Buffer(step.Scalar("float"), [D]))
    

class Expert1(step.Fn):
    def __init__(self, input, output):
        super().__init__("Expert1", input, output)
    
    def apply(self, input):
        return [input[0]]
        
    
fn_expert1 = Expert1(step.STuple((step.Buffer(step.Scalar("float"), [D]), step.Scalar("float"))), step.Buffer(step.Scalar("float"), [D]))
    

class Affinity(step.Fn):
    def __init__(self, input, output):
        super().__init__("Affinity", input, output)
    
    def apply(self, input):
        return [sigmoid(input[0] @ input_data['Wg'])]
        
    
fn_affinity = Affinity(step.Buffer(step.Scalar("float"), [D]), step.Scalar("float"))
    

class Sum(step.Fn):
    def __init__(self, input, output):
        super().__init__("Sum", input, output)

    def getInit(self):
        return [torch.zeros(D_value)]

    def apply(self, state, input):
        return [state[0] + input[0]]
        
    
fn_sum = Sum(step.Buffer(step.Scalar("float"), [D]), step.Buffer(step.Scalar("float"), [D]))
    

def prepare():
    E0 = step.Stream("E0", step.Scalar("float"), 2, [D, N, M])
    E0.ctx = ctx
    E0.data = [input_data['E0']]
    
    E1 = step.Stream("E1", step.Multihot(step.Scalar("float"), E), 1, [N, M])
    E1.ctx = ctx
    E1.data = [input_data['E1']]
    return E0, E1
def check_shape(S0):
    output_dtype_S0 = step.Buffer(step.Scalar("float"), [D])
    assert S0.dtype == step.Buffer(step.Scalar("float"), [D]), f"The output dtype should be {output_dtype_S0.dtype} but got {S0.dtype}"
    assert S0.shape == [N, M], f"The output shape should be [N, M] but got {S0.shape}"
    
def check_data(S0):
    E0_data = input_data['E0'] 
    score = input_data['E1'][:, :, 0].unsqueeze(-1)
    affinity = sigmoid(E0_data @ input_data['Wg']).unsqueeze(-1)
    S0_data_0 = (affinity * (gelu(E0_data @ input_data['W'])) + E0_data) * score + E0_data * (1 - score)
    torch.testing.assert_close(S0.data[0], S0_data_0)
    

def test():
    E0, E1 = prepare()
    S0 = body(E0, E1)
    check_shape(S0)
    check_data(S0)
    
