from reinforce.base import Storage
from reinforce.agents import QueryImpl, QueryPlan, QueryValue, QueryPyBody, QueryAffineImpl, QueryAffineImplTwoStages, QueryAffinePyBody, prepare_prompt_query_impl, prepare_prompt_query_impl_with_feedback, prepare_prompt_eliminate_identity, prepare_prompt_query_py_body, prepare_prompt_query_affine_rewrite, prepare_prompt_affine_impl_stage_0
from tools import yaml_to_code
import yaml
import json
import os
import subprocess
import re

def query_impl_once(key: str, config_dict: dict, config: Storage, storage: Storage):
    config.store(key, config_dict)
    agent = QueryImpl(key, config)
    with open(agent.config.retrieve(key)["task_path"], "r") as f:
        task_str = f.read()
        task_data = yaml.safe_load(task_str)
        storage.store(key, {"task": task_data})
    prompt = prepare_prompt_query_impl(key, config, storage)
    # Store the prompt to the temp_dir
    with open(os.path.join(agent.config.retrieve(key)["temp_dir"], "prompt.md"), "w") as f:
        f.write(prompt)
    agent.run(storage)
    agent.deduplicate_failure(storage)
    agent.deduplicate_success(storage)
    config.dump(os.path.join(agent.config.retrieve(key)["temp_dir"], "query_impl_config.yaml"))
    agent.log_yaml_list(storage.retrieve(key)["failure"], "failure_impl")
    agent.log_yaml_list(storage.retrieve(key)["success"], "success_impl")
    agent.log_yaml_list(storage.retrieve(key)["failure_rep"], "failure_impl_rep")
    agent.log_yaml_list(storage.retrieve(key)["success_rep"], "success_impl_rep")


def query_plan_once(key: str, config_dict: dict, config: Storage, storage: Storage):
    config.store(key, config_dict)
    agent = QueryPlan(key, config)
    with open(agent.config.retrieve(key)["task_path"], "r") as f:
        task_str = f.read()
        task_data = yaml.safe_load(task_str)
        storage.store(key, {"task": task_data})
    agent.run(storage)
    agent.top_kd_plan(storage)
    agent.log_yaml_list(storage.retrieve(key)["failure"], "failure_plan")
    agent.log_yaml_list(storage.retrieve(key)["success"], "success_plan")
    config.dump(os.path.join(agent.config.retrieve(key)["temp_dir"], "query_plan_config.yaml"))
    agent.log_yaml_list(storage.retrieve(key)["subtasks"], "subtask") # Necessary for the next step
    with open(os.path.join(agent.config.retrieve(key)["temp_dir"], "ids_dict.json"), "w") as f:
        json.dump(storage.retrieve(key)["ids_dict"], f)

def query_value_once(key: str, config_dict: dict, config: Storage, storage: Storage, ValueAgent=QueryValue):
    config.store(key, config_dict)
    agent = ValueAgent(key, config)
    agent.run(storage)
    config.dump(os.path.join(agent.config.retrieve(key)["temp_dir"], "config.yaml"))
    # Store the prompt to the temp_dir
    with open(os.path.join(agent.config.retrieve(key)["temp_dir"], "prompt.md"), "w") as f:
        f.write(storage.retrieve(key)["prompt"])
    # Store the feedback to the temp_dir
    with open(os.path.join(agent.config.retrieve(key)["temp_dir"], "full_feedback.yaml"), "w") as f:
        yaml.dump(storage.retrieve(key)["full_feedback"], f)
    # Store the feedback to the temp_dir
    with open(os.path.join(agent.config.retrieve(key)["temp_dir"], "feedback.yaml"), "w") as f:
        yaml.dump(storage.retrieve(key)["feedback"], f)

def query_subtask_impl_once(key: str, config_dict: dict, config: Storage, storage: Storage):
    config.store(key, config_dict)
    agent = QueryImpl(key, config)
    prompt = prepare_prompt_query_impl(key, config, storage)
    agent.run(storage)
    agent.deduplicate_failure(storage)
    config.dump(os.path.join(agent.config.retrieve(key)["temp_dir"], "query_impl_config.yaml"))
    agent.log_yaml_list(storage.retrieve(key)["failure"], "failure_impl")
    agent.log_yaml_list(storage.retrieve(key)["success"], "success_impl")
    agent.log_yaml_list(storage.retrieve(key)["failure_rep"], "failure_impl_rep")
    # Store the prompt to the temp_dir
    with open(os.path.join(agent.config.retrieve(key)["temp_dir"], "prompt.md"), "w") as f:
        f.write(prompt)

def query_impl_with_feedback_once(key: str, config_dict: dict, config: Storage, storage: Storage):
    config.store(key, config_dict)
    agent = QueryImpl(key, config)
    prompt = prepare_prompt_query_impl_with_feedback(key, config, storage)
    # Store the prompt to the temp_dir
    with open(os.path.join(agent.config.retrieve(key)["temp_dir"], "prompt.md"), "w") as f:
        f.write(prompt)
    agent.run(storage)
    agent.deduplicate_failure(storage)
    config.dump(os.path.join(agent.config.retrieve(key)["temp_dir"], "config.yaml"))
    agent.log_yaml_list(storage.retrieve(key)["failure"], "failure_impl")
    agent.log_yaml_list(storage.retrieve(key)["success"], "success_impl")
    agent.log_yaml_list(storage.retrieve(key)["failure_rep"], "failure_impl_rep")


def query_identity_elimination_once(key: str, config_dict: dict, config: Storage, storage: Storage):
    config.store(key, config_dict)
    agent = QueryImpl(key, config)
    prompt = prepare_prompt_eliminate_identity(key, config, storage)
    agent.run(storage)
    agent.deduplicate_success(storage)
    config.dump(os.path.join(agent.config.retrieve(key)["temp_dir"], "config.yaml"))
    agent.log_yaml_list(storage.retrieve(key)["failure"], "failure_impl")
    agent.log_yaml_list(storage.retrieve(key)["success"], "success_impl")
    agent.log_yaml_list(storage.retrieve(key)["success_rep"], "success_impl_rep")
    # Store the prompt to the temp_dir
    with open(os.path.join(agent.config.retrieve(key)["temp_dir"], "prompt.md"), "w") as f:
        f.write(prompt)

def query_py_body_once(key: str, config_dict: dict, config: Storage, storage: Storage):
    config.store(key, config_dict)
    agent = QueryPyBody(key, config)
    with open(agent.config.retrieve(key)["task_path"], "r") as f:
        task_str = f.read()
        storage.store(key, {"task": task_str})
    prompt = prepare_prompt_query_py_body(key, config, storage)
    with open(os.path.join(agent.config.retrieve(key)["temp_dir"], "prompt.md"), "w") as f:
        f.write(prompt)
    agent.run(storage)
    agent.deduplicate_failure(storage)
    agent.deduplicate_success(storage)
    config.dump(os.path.join(agent.config.retrieve(key)["temp_dir"], "config.yaml"))
    agent.log_py_list(storage.retrieve(key)["failure"], "failure_impl")
    agent.log_py_list(storage.retrieve(key)["success"], "success_impl")
    agent.log_py_list(storage.retrieve(key)["failure_rep"], "failure_impl_rep")
    agent.log_py_list(storage.retrieve(key)["success_rep"], "success_impl_rep")

def query_affine_py_body_once(key: str, config_dict: dict, config: Storage, storage: Storage):
    config.store(key, config_dict)
    agent = QueryAffinePyBody(key, config)
    with open(agent.config.retrieve(key)["task_path"], "r") as f:
        task_str = f.read()
        storage.store(key, {"task": task_str})
    prompt = prepare_prompt_query_py_body(key, config, storage)
    with open(os.path.join(agent.config.retrieve(key)["temp_dir"], "prompt.md"), "w") as f:
        f.write(prompt)
    agent.run(storage)
    agent.deduplicate_failure(storage)
    agent.deduplicate_success(storage)
    config.dump(os.path.join(agent.config.retrieve(key)["temp_dir"], "config.yaml"))
    agent.log_py_list(storage.retrieve(key)["failure"], "failure_impl")
    agent.log_py_list(storage.retrieve(key)["success"], "success_impl")
    agent.log_py_list(storage.retrieve(key)["failure_rep"], "failure_impl_rep")
    agent.log_py_list(storage.retrieve(key)["success_rep"], "success_impl_rep")

def query_affine_rewrite_once(key: str, config_dict: dict, config: Storage, storage: Storage):
    config.store(key, config_dict)
    agent = QueryAffineImpl(key, config)
    with open(agent.config.retrieve(key)["task_path"], "r") as f:
        task_str = f.read()
        task_data = yaml.safe_load(task_str)
    with open(agent.config.retrieve(key)["test_path"], "r") as f:
        test_str = f.read()
    storage.store(key, {"test": test_str, "task": task_data})
    prompt = prepare_prompt_query_affine_rewrite(key, config, storage)
    # Store the prompt to the temp_dir
    with open(os.path.join(agent.config.retrieve(key)["temp_dir"], "prompt.md"), "w") as f:
        f.write(prompt)
    agent.run(storage)
    agent.deduplicate_failure(storage)
    agent.deduplicate_success(storage)
    config.dump(os.path.join(agent.config.retrieve(key)["temp_dir"], "config.yaml"))
    agent.log_yaml_list(storage.retrieve(key)["failure"], "failure_impl")
    agent.log_yaml_list(storage.retrieve(key)["success"], "success_impl")
    agent.log_yaml_list(storage.retrieve(key)["failure_rep"], "failure_impl_rep")
    agent.log_yaml_list(storage.retrieve(key)["success_rep"], "success_impl_rep")

def query_affine_impl_once(key: str, config_dict: dict, config: Storage, storage: Storage):
    config.store(key, config_dict)
    agent = QueryAffineImpl(key, config)
    with open(agent.config.retrieve(key)["task_path"], "r") as f:
        task_str = f.read()
        task_data = yaml.safe_load(task_str)
        storage.store(key, {"task": task_data})
    prompt = prepare_prompt_query_impl(key, config, storage)
    # Store the prompt to the temp_dir
    with open(os.path.join(agent.config.retrieve(key)["temp_dir"], "prompt.md"), "w") as f:
        f.write(prompt)
    agent.run(storage)
    agent.deduplicate_failure(storage)
    agent.deduplicate_success(storage)
    config.dump(os.path.join(agent.config.retrieve(key)["temp_dir"], "query_impl_config.yaml"))
    agent.log_yaml_list(storage.retrieve(key)["failure"], "failure_impl")
    agent.log_yaml_list(storage.retrieve(key)["success"], "success_impl")
    agent.log_yaml_list(storage.retrieve(key)["failure_rep"], "failure_impl_rep")
    agent.log_yaml_list(storage.retrieve(key)["success_rep"], "success_impl_rep")

def query_affine_impl_two_stages(key: str, config_dict: dict, config: Storage, storage: Storage):
    config.store(key, config_dict)
    agent = QueryAffineImplTwoStages(key, config)
    with open(agent.config.retrieve(key)["task_path"], "r") as f:
        task_str = f.read()
        task_data = yaml.safe_load(task_str)
        storage.store(key, {"task": task_data})
    prompt = prepare_prompt_affine_impl_stage_0(key, config, storage)
    with open(os.path.join(agent.config.retrieve(key)["temp_dir"], "prompt_stage_0.md"), "w") as f:
        f.write(prompt)
    agent.run(storage)
    agent.deduplicate_failure(storage)
    agent.deduplicate_success(storage)
    config.dump(os.path.join(agent.config.retrieve(key)["temp_dir"], "config.yaml"))
    agent.log_yaml_list(storage.retrieve(key)["failure"], "failure_impl")
    agent.log_yaml_list(storage.retrieve(key)["success"], "success_impl")
    agent.log_yaml_list(storage.retrieve(key)["failure_rep"], "failure_impl_rep")
    agent.log_yaml_list(storage.retrieve(key)["success_rep"], "success_impl_rep")
    agent.log_md_list(storage.retrieve(key)["success_prompt_1"], "success_stage_1_prompt")
    agent.log_md_list(storage.retrieve(key)["failure_prompt_1"], "failure_stage_1_prompt")

def compose_example_with_helpers(ori_example_path, new_example_path, helpers_path, helper_task_path_list, helper_impl_path_list):
    # Assemble helpers
    helpers = {"helpers": []}
    for helper_task_path, helper_impl_path in zip(helper_task_path_list, helper_impl_path_list):
        helper_impl_data = yaml.safe_load(open(helper_impl_path, "r"))
        helper_task_data = yaml.safe_load(open(helper_task_path, "r"))
        helpers["helpers"].append(yaml_to_code.convert_to_literal({**helper_task_data, **helper_impl_data}))
    yaml.dump(helpers, open(helpers_path, "w"), default_flow_style=False, sort_keys=False, width=float("inf")) # Cannot use safe_dump, error: 'cannot represent an object'
    # Just concatenate the strings
    with open(ori_example_path, "r") as f:
        ori_example_str = f.read()
    with open(helpers_path, "r") as f:
        helpers_str = f.read()
    with open(new_example_path, "w") as f:
        f.write(ori_example_str + "\n\n" + helpers_str)
    return new_example_path

def check_ref_code(ref_code_list, input_code):
    input_code_list = input_code.split("\n")
    ref_code_exist = []
    for ref_code in ref_code_list:
        exist = False
        for input_segment in input_code_list:
            if ref_code in input_segment:
                exist = True
                break
        ref_code_exist.append(exist)
    if all(ref_code_exist):
        return True
    return False

def collect_error_to_files(id, config_data):
    # Check is the temp test file exists. If not, create it. Otherwise, read it.
    temp_dir = config_data["temp_dir"]
    model_name = config_data["model_name"]
    temp_test_path = os.path.join(temp_dir, f"test_{id}_{model_name}.py")
    if not os.path.exists(temp_test_path):
        print(f"{temp_test_path} does not exist")
    result = subprocess.run(
        ["pytest", temp_test_path, "--capture=no", "--no-header"], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True  # Automatically decode output
    )
    assert result.returncode != 0
    test_output = result.stdout
    # Extract the core error message
    output = []
    error_match = re.search(r'(E\s+.*?)(?=\n\n|$)', test_output, re.DOTALL)
    
    if error_match:
        error_msg = error_match.group(1).strip()
        # Split the error message into lines and only keep the first line
        error_msg = error_msg.split("\n")[0]
        # Remove the "E     " prefix
        error_msg = error_msg[8:]
        
        line_numbers = re.findall(rf'{temp_test_path}:(\d+):', test_output)
            
        # Convert to integers and find max
        if line_numbers:
            line_number = max(map(int, line_numbers))
        else:
            line_number = None
        
        # Format output
        if line_number:
            # Get the code at the line number in temp_test_path
            with open(temp_test_path, "r") as f:
                lines = f.readlines()
                code_string = lines[line_number - 1].strip()
                code = yaml_to_code.clean_python_code(code_string)
            output.append(f"{code}")
        output.append(f"{error_msg}")
        output.append("")

    return (id, output)