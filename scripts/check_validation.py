import pytest
import argparse
from tools import count_usage

def check_correctness_affine(py_file_path):
    result = pytest.main([py_file_path], plugins=[])
    if "test" in py_file_path:
        return result == pytest.ExitCode.OK
    usage_all_once = count_usage.check_affine_type(py_file_path)
    return usage_all_once and result == pytest.ExitCode.OK

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to the Python file to check")
    args = parser.parse_args()
    if check_correctness_affine(args.file):
        print("Correct.")
    else:
        print(f"{args.file} is incorrect!")