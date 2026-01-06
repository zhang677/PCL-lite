
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
}
input_data = {
    'E0': torch.rand(N_value, D_value),
}

class Add(step.Fn):
    def __init__(self, input, output):
        super().__init__("Add", input, output)

    def getInit(self):
        return [torch.tensor(0)]

    def apply(self, state, input):
        return [state[0] + input[0]]
        
    
fn_add = Add(step.Scalar("float"), step.Scalar("float"))
    

class Square(step.Fn):
    def __init__(self, input, output):
        super().__init__("Square", input, output)
    
    def apply(self, input):
        return [input[0] * input[0]]
        
    
fn_square = Square(step.Scalar("float"), step.Scalar("float"))
    

class DivByD(step.Fn):
    def __init__(self, input, output):
        super().__init__("DivByD", input, output)
    
    def apply(self, input):
        return [input[0] / D_value]
        
    
fn_divbyD = DivByD(step.Scalar("float"), step.Scalar("float"))
    

class Minus(step.Fn):
    def __init__(self, input, output):
        super().__init__("Minus", input, output)
    
    def apply(self, input):
        return [input[0] - input[1]]
        
    
fn_minus = Minus(step.STuple((step.Scalar("float"), step.Scalar("float"))), step.Scalar("float"))
    

class AddEps(step.Fn):
    def __init__(self, input, output):
        super().__init__("AddEps", input, output)
    
    def apply(self, input):
        return [input[0] + 1e-5]
        
    
fn_addeps = AddEps(step.Scalar("float"), step.Scalar("float"))
    

class Sqrt(step.Fn):
    def __init__(self, input, output):
        super().__init__("Sqrt", input, output)
    
    def apply(self, input):
        return [torch.sqrt(input[0])]
        
    
fn_sqrt = Sqrt(step.Scalar("float"), step.Scalar("float"))
    

class Div(step.Fn):
    def __init__(self, input, output):
        super().__init__("Div", input, output)
    
    def apply(self, input):
        return [input[0] / input[1]]
        
    
fn_div = Div(step.STuple((step.Scalar("float"), step.Scalar("float"))), step.Scalar("float"))
    

def prepare():
    E0 = step.Stream("E0", step.Scalar("float"), 1, [D, N])
    E0.ctx = ctx
    E0.data = [input_data['E0']]
    return E0
def check_shape(S0):
    output_dtype_S0 = step.Scalar("float")
    assert S0.dtype == step.Scalar("float"), f"The output dtype should be {output_dtype_S0.dtype} but got {S0.dtype}"
    assert S0.shape == [D, N], f"The output shape should be [D, N] but got {S0.shape}"
    
def check_data(S0):
    
    S0_data_0 = (input_data['E0'] - input_data['E0'].mean(dim=1, keepdim=True)) / torch.sqrt(input_data['E0'].var(dim=1, keepdim=True, unbiased=False) + 1e-5)
    torch.testing.assert_close(S0.data[0], S0_data_0)
    

def test():
    E0 = prepare()
    S0 = body(E0)
    check_shape(S0)
    check_data(S0)
    
