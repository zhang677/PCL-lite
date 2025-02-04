import ast
from typing import Optional
import copy
import argparse
from pathlib import Path
import shutil
import os
import json

class NameNormalizer(ast.NodeTransformer):
    """
    Normalizes variable names in AST to check structural equivalence
    regardless of variable names.
    """
    def __init__(self):
        self.name_map: dict[str, str] = {}
        self.counter = 0
        
    def _get_normalized_name(self, original: str) -> str:
        if original not in self.name_map:
            self.name_map[original] = f"VAR_{self.counter}"
            self.counter += 1
        return self.name_map[original]
    
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Normalize variable names"""
        normalized_name = self._get_normalized_name(node.id)
        return ast.Name(id=normalized_name, ctx=node.ctx)
    
    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Normalize function argument names"""
        normalized_name = self._get_normalized_name(node.arg)
        return ast.arg(arg=normalized_name, annotation=node.annotation)
    
    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        """Normalize string constants that might be variable names"""
        if isinstance(node.value, str) and node.value in self.name_map:
            return ast.Constant(value=self.name_map[node.value])
        return node

class ASTComparator:
    def __init__(self):
        self.ignore_fields = {'lineno', 'col_offset', 'end_lineno', 'end_col_offset', 'ctx'}
    
    def compare_nodes(self, node1: Optional[ast.AST], node2: Optional[ast.AST]) -> bool:
        """Compare two AST nodes for structural equality"""
        # Handle None cases
        if node1 is None and node2 is None:
            return True
        if node1 is None or node2 is None:
            return False
        
        # Check if nodes are of the same type
        if type(node1) != type(node2):
            return False
        
        # Get all relevant fields
        fields1 = {field: getattr(node1, field) 
                  for field in node1._fields 
                  if field not in self.ignore_fields}
        fields2 = {field: getattr(node2, field) 
                  for field in node2._fields 
                  if field not in self.ignore_fields}
        
        # Compare all fields
        if fields1.keys() != fields2.keys():
            return False
        
        for field in fields1:
            value1 = fields1[field]
            value2 = fields2[field]
            
            # Handle lists (like body of function)
            if isinstance(value1, list) and isinstance(value2, list):
                if len(value1) != len(value2):
                    return False
                for item1, item2 in zip(value1, value2):
                    if not self.compare_nodes(item1, item2):
                        return False
            # Handle nested AST nodes
            elif isinstance(value1, ast.AST) and isinstance(value2, ast.AST):
                if not self.compare_nodes(value1, value2):
                    return False
            # Handle primitive values
            else:
                if value1 != value2:
                    return False
        
        return True

def check_program_equivalence(program1: str, program2: str) -> bool:
    """
    Check if two programs are equivalent by comparing their normalized ASTs.
    """
    # Parse programs into ASTs
    try:
        ast1 = ast.parse(program1)
        ast2 = ast.parse(program2)
    except SyntaxError as e:
        print(f"Syntax error in one of the programs: {e}")
        return False
    
    # Create normalizers and normalize both ASTs
    normalizer1 = NameNormalizer()
    normalizer2 = NameNormalizer()
    
    normalized_ast1 = normalizer1.visit(copy.deepcopy(ast1))
    normalized_ast2 = normalizer2.visit(copy.deepcopy(ast2))
    
    # Compare the normalized ASTs
    comparator = ASTComparator()
    are_equivalent = comparator.compare_nodes(normalized_ast1, normalized_ast2)
    
    return are_equivalent

class EquivalentSet:
    def __init__(self, equiv_func):
        self._equiv_func = equiv_func
        self.item_map = {}
    
    def add(self, item, item_name):
        equal_key = None
        new_list = []
        for (k, v) in self.item_map.items():
            if self._equiv_func(item, k):
                new_list = v + [item_name]
                equal_key = k
                break
        if new_list:
            self.item_map[equal_key] = new_list
            return False
        else:
            self.item_map[item] = [item_name]
            return True


# Example usage
program1 = """
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
}
input_data = {
    'E0': torch.rand(K_value, M_value),
}

def test(): 

    O0_data_0 = input_data['E0'].unsqueeze(0).repeat(N_value, 1, 1)
 
    assert O0_data_0.shape == (N_value, K_value, M_value)
"""

program2 = """
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
}
input_data = {
    'E0': torch.rand(K_value, M_value),
}

def test(): 

    S0_data_0 = input_data['E0'].unsqueeze(0).repeat(N_value, 1, 1)
 
    assert S0_data_0.shape == (N_value, K_value, M_value)
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Check AST equivalence of two Python programs.')
    parser.add_argument("--mode", type=str, default="single")
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--p1_path", type=str, default=".")
    parser.add_argument("--p2_path", type=str, default=".")
    args = parser.parse_args()
    if args.mode == "single":
        print("Checking AST equivalence...")
        are_equivalent = check_program_equivalence(program1, program2)
        # Print the name mappings for debugging
        if are_equivalent:
            print("\nPrograms are structurally equivalent!")
        else:
            print("\nPrograms are not structurally equivalent.")
            
            # Additional debugging information
            print("\nKey differences may include:")
            print("- Different tensor operations (permute vs no permute)")
            print("- Different tensor shapes in torch.rand() calls")
            print("- Different assertion conditions")
        
        code_history = EquivalentSet(check_program_equivalence)
        print(code_history.add(program1))
        print(code_history.add(program2))
        print(len(code_history))
    elif args.mode == "batch":
        code_history = EquivalentSet(check_program_equivalence)
        input_path = Path(args.input_dir)
        py_files = list(input_path.glob("*.py"))
        os.makedirs(args.output_dir, exist_ok=True)
        subtask_counter = 0
        for py_file in py_files:
            print(f"Processing {py_file}")
            yaml_file = py_file.with_suffix(".yaml")
            if code_history.add(py_file.read_text(), py_file.stem):
                # rename the prefix to subtask_{num_unique}
                new_py_name = f"subtask_{subtask_counter}.py"
                new_yaml_name = f"subtask_{subtask_counter}.yaml"
                new_py_path = Path(args.output_dir) / new_py_name
                new_yaml_path = Path(args.output_dir) / new_yaml_name
                shutil.copy(py_file, new_py_path)
                shutil.copy(yaml_file, new_yaml_path)
                subtask_counter += 1
        code_history_path = Path(args.output_dir) / "statistics.json"
        code_history_config = {}
        for (id, (k, v)) in enumerate(code_history.item_map.items()):
            code_history_config[f"subtask_{id}"] = v
        with open(code_history_path, "w") as file:
            json.dump(code_history_config, file, indent=4)
        # Print the length of each equivalence class
        for (k, v) in code_history.item_map.items():
            print(f"{k}: {len(v)}")
    elif args.mode == "compare":
        program1 = Path(args.p1_path).read_text()
        program2 = Path(args.p2_path).read_text()
        are_equivalent = check_program_equivalence(program1, program2)
        if are_equivalent:
            print("\nPrograms are structurally equivalent!")
        else:
            print("\nPrograms are not structurally equivalent.")