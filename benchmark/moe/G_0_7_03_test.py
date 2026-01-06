
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


sigmoid = torch.nn.Sigmoid()


input_dtype = {
    'E0': step.Scalar("float"),
}
input_data = {
    'E0': torch.randn(M_value, K_value),
    'Wg': torch.randn(K_value),
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
    

def prepare():
    E0 = step.Stream("E0", step.Scalar("float"), 1, [K, M])
    E0.ctx = ctx
    E0.data = [input_data['E0']]
    return E0
def check_shape(S0, S1):
    output_dtype_S0 = step.Scalar("float")
    assert S0.dtype == step.Scalar("float"), f"The output dtype should be {output_dtype_S0.dtype} but got {S0.dtype}"
    assert S0.shape == [M], f"The output shape should be [M] but got {S0.shape}"
    
    output_dtype_S1 = step.Scalar("float")
    assert S1.dtype == step.Scalar("float"), f"The output dtype should be {output_dtype_S1.dtype} but got {S1.dtype}"
    assert S1.shape == [M], f"The output shape should be [M] but got {S1.shape}"
    
def check_data(S0, S1):
    affinity = sigmoid(input_data['E0'] @ input_data['Wg'])
    _, indices = torch.topk(affinity, 3, dim=0)
    multihot = torch.zeros_like(affinity, dtype=torch.float32)
    S0_data_0 = multihot.scatter(0, indices, 1.0)
    torch.testing.assert_close(S0.data[0], S0_data_0)
    
    
    S1_data_0 = affinity
    torch.testing.assert_close(S1.data[0], S1_data_0)
    

def test():
    E0 = prepare()
    S0, S1 = body(E0)
    check_shape(S0, S1)
    check_data(S0, S1)
    
