import yaml
import re
import argparse
import os
from functools import reduce

prefix = """
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
"""

def replace_one_with_str(dims):
    if isinstance(dims, list):
        return list(map(lambda x: str(x) if x == 1 else x, dims))
    elif dims == 1:
        return "1"
    elif isinstance(dims, str):
        return dims
    else:
        raise ValueError(f"Unknown dims: {dims}")
    

def extract_explicit_dtype(dtype):
    if isinstance(dtype, list):
        subtypes = list(map(extract_explicit_dtype, dtype))
        return f"step.STuple(({', '.join(subtypes)}))"
    elif isinstance(dtype, str):
        dtype = dtype.replace("fp32", "step.Scalar(\"float\")")
        dtype = dtype.replace("Buffer", "step.Buffer")
        dtype = dtype.replace("Multihot", "step.Multihot")
        return dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

def extract_fn(fn):
    name = fn.get("name", "")
    func_name = fn.get("func_name", "")
    apply = fn.get("apply", "")
    input_dtype = fn.get("input_dtype", "")
    output_dtype = fn.get("output_dtype", "")
    init_list = fn.get("init", [])
    if apply:
        if init_list:
            return extract_fn_with_init(name, func_name, apply, input_dtype, output_dtype, init_list)
        else:
            return extract_fn_wo_init(name, func_name, apply, input_dtype, output_dtype)
    else:
        raise ValueError(f"Missing apply function for {name}")

def insert_indent(line_list, indent):
    # Add indent to each "\n" in line_list
    return line_list.replace("\n", indent)

def extract_fn_with_init(name, func_name, apply, input_dtype, output_dtype, init_list):
    init_returns = []
    if not isinstance(output_dtype, list):
        output_dtype_list = [output_dtype]
    else:
        output_dtype_list = output_dtype
    for (dtype, value) in zip(output_dtype_list, init_list):
        if "Buffer" in dtype:
            # Extract the [M, N, ...] from dtype_code
            # Cannot handle shape 1 for now
            match = re.search(r'\[(.*?)\]', dtype)
            if match:
                full_str = match.group(1)
                buff_dims = full_str.split(", ")
                base_str = "(" + ", ".join(map(lambda x: x+"_value", reversed(buff_dims))) + ")"
                if value == 0:
                    init_returns.append(f"torch.zeros{base_str}")
                elif value == 1:
                    init_returns.append(f"torch.ones{base_str}")
                else:
                    raise ValueError(f" Unsupported Buffer init value {value}.")
            else:
                raise ValueError("Input string does not contain content in square brackets.")
        elif dtype == "fp32":
            if value == 0:
                init_returns.append("torch.tensor(0)")
            elif value == 1:
                init_returns.append("torch.tensor(1)")
            elif value == -1:
                init_returns.append("torch.tensor(-1)")
            elif value == "-inf":
                init_returns.append("torch.tensor(float('-inf'))")
            else:
                raise ValueError(f" Unsupported Scalar init value {value}.")
        else:
            raise ValueError(f" Unsupported dtype {dtype}.")
    
    class_str = f"""
class {name}(step.Fn):
    def __init__(self, input, output):
        super().__init__("{name}", input, output)

    def getInit(self):
        return [{', '.join(init_returns)}]

    def apply(self, state, input):
        {insert_indent(apply, "\n        ")}
    """
    obj_str = f"""
{func_name} = {name}({extract_explicit_dtype(input_dtype)}, {extract_explicit_dtype(output_dtype)})
    """
    return class_str + obj_str + "\n"

def extract_fn_wo_init(name, func_name, apply, input_dtype, output_dtype):
    class_str = f"""
class {name}(step.Fn):
    def __init__(self, input, output):
        super().__init__("{name}", input, output)
    
    def apply(self, input):
        {insert_indent(apply, "\n        ")}
    """
    obj_str = f"""
{func_name} = {name}({extract_explicit_dtype(input_dtype)}, {extract_explicit_dtype(output_dtype)})
    """
    return class_str + obj_str + "\n"

def dims_to_datadims(dims, dtype_code):
    if "Buffer" in dtype_code:
        # Extract the [M, N, ...] from dtype_code
        match = re.search(r'\[(.*?)\]', dtype_code)
        if match:
            full_str = match.group(1)
            buff_dims = full_str.split(", ")
            data_dims = reversed(buff_dims + dims)
        else:
            raise ValueError("Input string does not contain content in square brackets.")
    else:
        data_dims = reversed(dims)
    return data_dims

def torch_data_init(data_gen, data_dims, input=None):
    if data_gen == "torch.rand" or data_gen == "torch.randn":
        return data_gen + "(" + ", ".join(map(lambda x: x+"_value", data_dims)) + ")"
    elif data_gen == "torch.ones":
        return data_gen + "((" + ", ".join(map(lambda x: x+"_value", data_dims)) + "), dtype=torch.float)"
    elif data_gen == "binary":
        return f"generate_binary_tensor(({', '.join(map(lambda x: x+'_value', data_dims))}))"
    else:
        dtype = input.get("dtype", {})
        assert "Multihot" in dtype, "Only support Multihot data_gen for now."
        # Extract the E from dtype
        match = re.search(r'Multihot\((\w+),\s*(\w+)\)', dtype)
        if match:
            scalar_dtype = match.group(1)
            assert scalar_dtype == "fp32", "Only support float for now."
            num_classes_symbol = match.group(2)
            return f"generate_multi_hot(({', '.join(map(lambda x: x+'_value', data_dims))}), {input["min"]}, {input["max"]}, {num_classes_symbol}_value)"
        else:
            raise ValueError("Input string cannot be decoded as Multihot.")


def extract_func_lines(func):
    lines = func.split('\n')
    id = len(lines) - 1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            id = i
            break
    intermediate_lines = lines[:id]
    result_line = lines[id]
    return intermediate_lines, result_line


def yaml_to_code(data):

    inputs = data.get("inputs", [])
    data_dict = {}
    dtype_dict= {}
    input_stream_dict = {}
    input_names = []
    for input in inputs:
        name = input.get("name", '')
        dtype = input.get("dtype", {})
        dtype_code = extract_explicit_dtype(dtype)
        dtype_dict[name] = dtype_code
        dims = replace_one_with_str(input.get("dims", []))
        data_gen = input.get("data_gen", "")
        data_dims = dims_to_datadims(dims, dtype_code)
        data_dict[name] = torch_data_init(data_gen, data_dims, input)
        input_stream_dict[name] = f"""
    {name} = step.Stream(\"{name}\", {dtype_code}, {len(dims) - 1}, [{", ".join(dims)}])
    {name}.ctx = ctx
    {name}.data = [input_data['{name}']]
    """
        input_names.append(name)
    
    param_dict = {}
    parameters = data.get("parameters", [])
    for param in parameters:
        name = param.get("name", '')
        dtype = param.get("dtype", {})
        dtype_code = extract_explicit_dtype(dtype)
        dims = replace_one_with_str(param.get("dims", []))
        data_gen = param.get("data_gen", "")
        data_dims = dims_to_datadims(dims, dtype_code)
        param_dict[name] = torch_data_init(data_gen, data_dims)
    
    outputs = data.get("outputs", [])
    check_shape_str = ""
    check_data_str = ""
    output_names = []
    for output in outputs:
        name = output.get("name", '')
        dtype = output.get("dtype", '')
        dtype_code = extract_explicit_dtype(dtype)
        dims = replace_one_with_str(output.get("dims", []))
        check_shape_dtype = f"""
    output_dtype_{name} = {dtype_code}
    assert {name}.dtype == {dtype_code}, f"The output dtype should be {{output_dtype_{name}.dtype}} but got {{{name}.dtype}}"
    assert {name}.shape == [{', '.join(dims)}], f"The output shape should be [{', '.join(dims)}] but got {{{name}.shape}}"
    """
        check_data = ""
        for (i, func) in enumerate(output.get("data_transform", "")):
            intermediate_lines, result_line = extract_func_lines(func)
            check_data += f"""
    {insert_indent('\n'.join(intermediate_lines), "\n    ")}
    {name}_data_{i} = {result_line}
    torch.testing.assert_close({name}.data[{i}], {name}_data_{i})
    """
        check_shape_str += check_shape_dtype
        check_data_str += check_data
        output_names.append(name)

    listof_input_names = ", ".join(input_names)
    listof_output_names = ", ".join(output_names)
    check_shape_str = "def check_shape(" + listof_output_names + "):" + check_shape_str
    check_data_str = "def check_data(" + listof_output_names + "):" + check_data_str

    fn_str = ""
    fns = data.get("fns", [])
    for fn in fns:
        fn_str += extract_fn(fn)

    global_stmts = data.get("global", "")
    if global_stmts:
        global_stmts_str = f"""
{insert_indent(global_stmts, "\n")}
"""
    else:
        global_stmts_str = ""

    dtype_dict_str = "input_dtype = {\n"
    for key, value in dtype_dict.items():
        dtype_dict_str += f"    \'{key}\': {value},\n"
    dtype_dict_str += "}"

    data_dict_str = "input_data = {\n"
    for key, value in data_dict.items():
        data_dict_str += f"    \'{key}\': {value},\n"
    for key, value in param_dict.items():
        data_dict_str += f"    \'{key}\': {value},\n"
    data_dict_str += "}"

    prepare_str = "def prepare():"
    for name in input_names:
        prepare_str += input_stream_dict[name]
    prepare_str += "return " + listof_input_names

    test_str = f"""
def test():
    {listof_input_names} = prepare()
    {listof_output_names} = body({listof_input_names})
    check_shape({listof_output_names})
    check_data({listof_output_names})
    """

    impl = data.get("impl", "")
    if impl:
        impl_str = f"""
def body({listof_input_names}):
    {insert_indent(impl, "\n    ")}
"""
    else:
        impl_str = ""
    
    # Return the code
    return reduce(lambda x, y: x + "\n" + y, [prefix, 
                                            global_stmts_str,
                                            dtype_dict_str, 
                                            data_dict_str, 
                                            fn_str, 
                                            prepare_str,
                                            check_shape_str, 
                                            check_data_str, 
                                            test_str, 
                                            impl_str])

def decompose_step_yaml_to_code(input_file, output_dir):
    """
    Extract examples from a YAML file and save them as separate files.
    
    Args:
        input_file (str): Path to input YAML file
        output_dir (str): Directory to save extracted examples
    """
    with open(input_file, 'r') as f:
        data = yaml.safe_load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ops = data.get('ops', [])
    patterns = data.get('patterns', [])
    ops = ops + patterns
    for op in ops:
        op_name = op.get('name', 'unknown_op')
        examples = op.get('examples', [])
        for idx, example in enumerate(examples):
            # Prepare the output filename
            op_name = op_name.replace(' ', '_')
            output_filename = f"{op_name}_example_{idx+1}.py"
            output_path = os.path.join(output_dir, output_filename)

            # Write the example to the file
            with open(output_path, 'w') as f:
                f.write(yaml_to_code(example))
            print(f"Extracted example {idx+1} for op '{op_name}' to {output_path}")
    helpers = data.get('helpers', [])
    for (idx, helper) in enumerate(helpers):
        output_filename = f"helper_{idx+1}.py"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w') as f:
            f.write(yaml_to_code(helper))
        print(f"Extracted helper {idx+1} to {output_path}")

def literal_presenter(dumper, data):
    """Present multiline strings as literal blocks."""
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

def flow_list_presenter(dumper, data):
    """Present lists in flow style [x, y, z]."""
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

class literal(str): pass
class flow_list(list): pass

yaml.add_representer(literal, literal_presenter)
yaml.add_representer(flow_list, flow_list_presenter)

def should_be_flow_list(data):
    """Check if a list should use flow style."""
    if not isinstance(data, list):
        return False
    # Check if it's a simple list (numbers or single-line strings)
    return all(
        isinstance(x, (int, float)) or 
        (isinstance(x, str) and '\n' not in x and not isinstance(x, literal))
        for x in data
    )

def convert_to_literal(data):
    """Convert string values to literal blocks if they contain newlines."""
    if isinstance(data, dict):
        return {key: convert_to_literal(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Convert each item in the list
        converted_list = [convert_to_literal(item) for item in data]
        # Check if any item in the converted list is a literal
        has_literal = any(isinstance(x, literal) for x in converted_list)
        # If the list should be flow style and doesn't contain literals, make it a flow_list
        if should_be_flow_list(data) and not has_literal:
            return flow_list(converted_list)
        return converted_list
    elif isinstance(data, str):
        # Convert to literal if contains newlines or is indented
        if '\n' in data or 'input_data' in data:
            return literal(data)
    return data


def decompose_step_yaml(input_file, output_dir):

    with open(input_file, 'r') as f:
        data = yaml.safe_load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ops = data.get('ops', [])
    for op in ops:
        op_name = op.get('name', 'unknown_op')
        examples = op.get('examples', [])
        for idx, example in enumerate(examples):
            # Prepare the output filename
            op_name = op_name.replace(' ', '_')
            output_filename = f"{op_name}_example_{idx+1}.yaml"
            output_path = os.path.join(output_dir, output_filename)
            example_dict = {'examples': [convert_to_literal(example)]}

            # Write the example to the file
            with open(output_path, 'w') as f:
                yaml.dump(example_dict, f, default_flow_style=False, sort_keys=False, width=float("inf"))
            print(f"Extracted example {idx+1} for op '{op_name}' to {output_path}")

    helpers = data.get('helpers', [])
    for (idx, helper) in enumerate(helpers):
        output_filename = f"helper_{idx+1}.yaml"
        output_path = os.path.join(output_dir, output_filename)
        helper_dict = {'helpers': [convert_to_literal(helper)]}
        with open(output_path, 'w') as f:
            yaml.dump(helper_dict, f, default_flow_style=False, sort_keys=False, width=float("inf"))
        print(f"Extracted helper {idx+1} to {output_path}")

def yaml_plan_to_code(data):

    inputs = data.get("inputs", [])
    data_dict = {}
    dtype_dict= {}
    input_stream_dict = {}
    input_names = []
    for input in inputs:
        name = input.get("name", '')
        dtype = input.get("dtype", {})
        dtype_code = extract_explicit_dtype(dtype)
        dtype_dict[name] = dtype_code
        dims = replace_one_with_str(input.get("dims", []))
        data_gen = input.get("data_gen", "")
        data_dims = dims_to_datadims(dims, dtype_code)
        data_dict[name] = torch_data_init(data_gen, data_dims, input)
        input_stream_dict[name] = f"""
    {name} = step.Stream(\"{name}\", {dtype_code}, {len(dims) - 1}, [{", ".join(dims)}])
    {name}.ctx = ctx
    {name}.data = [input_data['{name}']]
    """
        input_names.append(name)

    param_dict = {}
    parameters = data.get("parameters", [])
    for param in parameters:
        name = param.get("name", '')
        dtype = param.get("dtype", {})
        dtype_code = extract_explicit_dtype(dtype)
        dims = replace_one_with_str(param.get("dims", []))
        data_gen = param.get("data_gen", "")
        data_dims = dims_to_datadims(dims, dtype_code)
        param_dict[name] = torch_data_init(data_gen, data_dims)

    outputs = data.get("outputs", [])
    check_data_shape_str = ""
    output_names = []
    for output in outputs:
        name = output.get("name", '')
        dtype = output.get("dtype", '')
        dtype_code = extract_explicit_dtype(dtype)
        dims = replace_one_with_str(output.get("dims", []))
        data_dims = []
        if isinstance(dtype, list):
            for single_dtype in dtype:
                data_dims.append(dims_to_datadims(dims, single_dtype))
        else:
            data_dims.append(dims_to_datadims(dims, dtype))
        check_data_shape = ""
        for (i, func) in enumerate(output.get("data_transform", "")):
            intermediate_lines, result_line = extract_func_lines(func)
            check_data_shape += f"""
    {insert_indent('\n'.join(intermediate_lines), "\n    ")}
    {name}_data_{i} = {result_line} 
    assert {name}_data_{i}.shape == ({', '.join(map(lambda x: x+"_value" if x != "1" else "1", data_dims[i]))})
    """
        check_data_shape_str += check_data_shape
        output_names.append(name)

    check_data_shape_str = "def test(): \n" + check_data_shape_str

    fn_str = ""
    fns = data.get("fns", [])
    for fn in fns:
        fn_str += extract_fn(fn)

    dtype_dict_str = "input_dtype = {\n"
    for key, value in dtype_dict.items():
        dtype_dict_str += f"    \'{key}\': {value},\n"
    dtype_dict_str += "}"

    data_dict_str = "input_data = {\n"
    for key, value in data_dict.items():
        data_dict_str += f"    \'{key}\': {value},\n"
    data_dict_str += "}"

    # Return the code
    return reduce(lambda x, y: x + "\n" + y, [prefix, 
                                            dtype_dict_str, 
                                            data_dict_str, 
                                            fn_str,
                                            check_data_shape_str])

def clean_python_code(code_string):
    """
    Remove comments and extra newlines from Python code while preserving code functionality.
    
    Args:
        code_string (str): Input Python code as a string
        
    Returns:
        str: Cleaned Python code with comments and extra newlines removed
    """
    # Split the code into lines
    lines = code_string.split('\n')
    
    # Process each line
    cleaned_lines = []
    for line in lines:
        # Remove leading and trailing whitespace
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
            
        # Skip comment-only lines
        if stripped.startswith('#'):
            continue
            
        # Remove inline comments while preserving string literals
        result = ''
        in_string = False
        string_char = None
        i = 0
        
        while i < len(line):
            char = line[i]
            
            # Handle string literals
            if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                result += char
                
            # Handle comments
            elif char == '#' and not in_string:
                break
                
            # Handle all other characters
            else:
                result += char
            
            i += 1
            
        # Add non-empty lines to result
        if result.strip():
            cleaned_lines.append(result.rstrip())
    
    # Join lines with single newlines
    return '\n'.join(cleaned_lines)

def batch_yaml_to_code(task_data, temp_dir, model_name, rounds, prefix):
    for id in range(rounds):
        temp_test_path = os.path.join(temp_dir, f"test_{id}_{model_name}.py")
        impl_path = os.path.join(temp_dir, f"{prefix}_{id}.yaml")
        if not os.path.exists(impl_path):
            continue
        with open(impl_path, "r") as f:
            impl_str = f.read()
            impl_data = yaml.safe_load(impl_str)
        data = {**task_data, **impl_data}
        code = yaml_to_code(data)
        with open(temp_test_path, 'w') as file:
            file.write(code)

def remove_all_py(temp_dir):
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".py"):
                os.remove(os.path.join(root, file))

def clean_model_name(model_name):
    return model_name.replace(":", "")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", help="The yaml file to convert")
    parser.add_argument("--output", help="The output file to write to")
    parser.add_argument("--mode", help="Mode to use", default="single")
    args = parser.parse_args()

    if args.mode == "single":
        with open(args.yaml, "r") as file:
            yaml_content = file.read()
        data = yaml.safe_load(yaml_content)
        with open(args.output, "w") as file:
            file.write(yaml_to_code(data))
    elif args.mode == "decode":
        decompose_step_yaml_to_code(args.yaml, args.output)
    elif args.mode == "deyaml":
        decompose_step_yaml(args.yaml, args.output)
    elif args.mode == "plan":
        with open(args.yaml, "r") as file:
            yaml_content = file.read()
        data = yaml.safe_load(yaml_content)
        with open(args.output, "w") as file:
            file.write(yaml_plan_to_code(data))
    else:
        raise ValueError(f"Unknown mode: {args.mode}")