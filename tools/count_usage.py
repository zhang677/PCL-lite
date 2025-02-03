import ast
import re
import argparse

class IterationTracker:
    def __init__(self):
        self.var_origins = {}  # Maps iteration variables to their source

    def track_iteration(self, target, iterable):
        if isinstance(target, ast.Name) and isinstance(iterable, ast.Name):
            self.var_origins[target.id] = iterable.id
        elif isinstance(target, ast.Tuple):
            if isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Name) and iterable.func.id == 'zip':
                for t, source in zip(target.elts, iterable.args):
                    if isinstance(t, ast.Name) and isinstance(source, ast.Name):
                        self.var_origins[t.id] = source.id

class ApplyCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.apply_counts = {}
        self.tracker = IterationTracker()
        
    def visit_For(self, node):
        self.tracker.track_iteration(node.target, node.iter)
        self.generic_visit(node)
        
    def visit_ListComp(self, node):
        # Handle list comprehension generators
        for generator in node.generators:
            self.tracker.track_iteration(generator.target, generator.iter)
        self.generic_visit(node)
        
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'apply':
            for arg in node.args:
                self.count_argument(arg)
        self.generic_visit(node)
    
    def count_argument(self, arg):
        if isinstance(arg, ast.Name):
            var_name = arg.id
            # If it's an iteration variable, count its source instead
            if var_name in self.tracker.var_origins:
                var_name = self.tracker.var_origins[var_name]
            self.apply_counts[var_name] = self.apply_counts.get(var_name, 0) + 1
        elif isinstance(arg, ast.Tuple):
            for elt in arg.elts:
                self.count_argument(elt)

def count_apply_usage(code_str):
    tree = ast.parse(code_str)
    visitor = ApplyCallVisitor()
    visitor.visit(tree)
    return visitor.apply_counts

def extract_body_function(file_content):
    pattern = r'def body\([^)]*\):'
    match = re.search(pattern, file_content)
    
    if not match:
        return None
        
    start = match.start()
    end = file_content.find('\n\n', start)
    if end == -1:
        end = len(file_content)
        
    return file_content[start:end]

def check_affine_type(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    body_function = extract_body_function(content)
    counts = count_apply_usage(body_function)
    return all(count == 1 for count in counts.values())

# Main execution
def main(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    body_function = extract_body_function(content)
    counts = count_apply_usage(body_function)

    # Print results in sorted order
    print("Number of times each variable is used in apply() calls:")
    for var, count in counts.items():
        print(f"{var}: {count} times")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path to the file to analyze")
    args = parser.parse_args()
    file_path = args.file_path
    main(file_path)
    print("All variables are used exactly once in apply() calls:", check_affine_type(file_path))