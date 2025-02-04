import os
import time
import yaml
import pytest
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tools.utils import query_anthropic, extract_code, query, clean_yaml, clean_code, query_text, print_elapsed_time
from tools import yaml_to_code, ast_analysis, count_usage
from reinforce.base import Agent, Storage

def prepare_prompt_query_impl(key, config: Storage, storage: Storage):
    config_data = config.retrieve(key)
    storage_data = storage.retrieve(key)
    example_path = config_data["example_path"]
    with open(example_path, "r") as f:
        example = f.read()
        storage_data["example"] = example
    task_data = storage_data["task"]
    task = yaml.dump(yaml_to_code.convert_to_literal(task_data), default_flow_style=False, sort_keys=False, width=float("inf"))
    prompt = f"""
```
{example}
```
Given the above example, complete the `impl` for the test
```
{task}
```
Please output the code in this format:   
```
impl: |-
``` 
"""     
    storage_data["prompt"] = prompt
    return prompt

def prepare_prompt_query_impl_with_feedback(key, config: Storage, storage: Storage):
    config_data = config.retrieve(key)
    storage_data = storage.retrieve(key)
    example_path = config_data["example_path"]
    with open(example_path, "r") as f:
        example = f.read()
        storage_data["example"] = example
    task_data = storage_data["task"]
    feedback_data = storage_data["feedback"]
    task = yaml.dump(yaml_to_code.convert_to_literal(task_data), default_flow_style=False, sort_keys=False, width=float("inf"))
    feedback = yaml.dump(yaml_to_code.convert_to_literal(feedback_data), default_flow_style=False, sort_keys=False, width=float("inf"))
    prompt = f"""
```
{example}
```
Given the above example, complete the `impl` for the test
```
{task}

Feedback:
{feedback}
```
Please output the code in this format:   
```
impl: |-
``` 
"""     
    storage_data["prompt"] = prompt
    return prompt

def prepare_prompt_eliminate_identity(key, config: Storage, storage: Storage):
    storage_data = storage.retrieve(key)
    ex_impl_data = storage_data["ex_impl"]
    ex_impl = yaml.dump(yaml_to_code.convert_to_literal(ex_impl_data), default_flow_style=False, sort_keys=False, width=float("inf"))
    prompt = f"""

This program pattern:
```
Ex = ...
Ey = step.Bufferize(a=?).apply(Ex)
Ez = step.Streamify().apply(Ey)
``` can be simplified to:
```
Ez = ...
```
because Bufferize followed by a Streamify is an identity operation when Ex's element type is `step.Scalar`. 
Please detect this pattern in the below impl and simplify it.
```
{ex_impl}
```
Please output the code in this format:   
```
impl: |-
``` 
"""     
    storage_data["prompt"] = prompt
    return prompt

def single_query_impl(id, prompt, task_data, config_data):
    temp_dir = config_data["temp_dir"]
    model_name = config_data["model_name"]
    temperature = config_data["temperature"]
    max_tokens = config_data["max_tokens"]
    system_prompt = config_data["system_prompt"]
    temp = {"usage": {}}
    response = query_text(prompt, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens, storage=temp)
    impl = extract_code(response)
    impl = clean_yaml(impl)
    temp_test_path = os.path.join(temp_dir, f"test_{id}_{yaml_to_code.clean_model_name(model_name)}.py")
    impl_data = yaml.safe_load(impl)
    data = {**task_data, **impl_data}
    code = yaml_to_code.yaml_to_code(data)
    with open(temp_test_path, 'w') as file:
        file.write(code)
    result = pytest.main([temp_test_path], plugins=[])
    os.remove(temp_test_path) # Comment this line to keep the test file when debugging
    return (result, impl_data, temp["usage"])

class QueryImpl(Agent):
    def __init__(self, key, config: Storage):
        super().__init__(key, config)
    
    def run(self, storage: Storage):
        config_data = self.config.retrieve(self.key)
        num_samples = config_data["num_samples"]
        num_workers = config_data["num_workers"]
        storage_data = storage.retrieve(self.key)
        task_data = storage_data["task"]        
        prompt = storage_data["prompt"] # self.prepare_prompt(storage)
        success = []
        failure = []
        usages = []
        start = time.time()
        with tqdm(total=num_samples, smoothing=0) as pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        single_query_impl,
                        id,
                        prompt,
                        task_data,
                        config_data
                    )
                    for id in range(num_samples)
                }
                for future in as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            if result[0] == pytest.ExitCode.OK:
                                success.append(result[1])
                            else:
                                failure.append(result[1])
                            usages.append(result[2])
                    except Exception as e:
                        print("Got an error!", e)
                        continue
        end = time.time()
        print_elapsed_time(start, end)
        storage_data["success"] = success
        storage_data["failure"] = failure # Stores a list of failed impls (not task+impl)
        storage_data["usage"] = usages
    
    def deduplicate_failure(self, storage: Storage):
        storage_data = storage.retrieve(self.key)
        failure = storage_data["failure"]
        task_data = storage_data["task"]
        code_history = ast_analysis.EquivalentSet(ast_analysis.check_program_equivalence)
        rep_list = []
        rep_impl_list = []
        for (id, impl_data) in enumerate(failure):
            data = {**task_data, **impl_data}
            code = yaml_to_code.yaml_to_code(data)
            code_history.add(code, id)
        for (_, v) in code_history.item_map.items():
            rep_list.append({**failure[v[0]]})
            rep_impl_list.append(failure[v[0]])
        storage_data["failure_rep"] = rep_list
        storage_data["failure_rep_impl"] = rep_impl_list

    def deduplicate_success(self, storage: Storage):
        storage_data = storage.retrieve(self.key)
        failure = storage_data["success"]
        task_data = storage_data["task"]
        code_history = ast_analysis.EquivalentSet(ast_analysis.check_program_equivalence)
        rep_list = []
        rep_impl_list = []
        for (id, impl_data) in enumerate(failure):
            data = {**task_data, **impl_data}
            code = yaml_to_code.yaml_to_code(data)
            code_history.add(code, id)
        for (_, v) in code_history.item_map.items():
            rep_list.append({**failure[v[0]]})
            rep_impl_list.append(failure[v[0]])
        storage_data["success_rep"] = rep_list
        storage_data["success_rep_impl"] = rep_impl_list

def prepare_prompt_query_py_body(key, config: Storage, storage: Storage):
    config_data = config.retrieve(key)
    storage_data = storage.retrieve(key)
    example_path = config_data["example_path"]
    with open(example_path, "r") as f:
        example = f.read()
        storage_data["example"] = example
    task_data = storage_data["task"]
    prompt = f"""
```
{example}
```
Ops define all the primitive you can use. Please implement the `body` function for the test.
```
{task_data}
```
Please output the code in this format:   
```
def body
``` 
"""     
    storage_data["prompt"] = prompt
    return prompt

def single_query_py_body(id, prompt, task_str, config_data):
    temp_dir = config_data["temp_dir"]
    model_name = config_data["model_name"]
    temperature = config_data["temperature"]
    max_tokens = config_data["max_tokens"]
    system_prompt = config_data["system_prompt"]
    temp = {"usage": {}}
    if "claude" in model_name:
        response = query_anthropic(prompt, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens, storage=temp)
        impl = extract_code(response['text'])
    else:
        response = query(prompt, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens, storage=temp)
        impl = extract_code(response)
    impl_str = clean_code(impl)
    temp_test_path = os.path.join(temp_dir, f"test_{id}_{model_name}.py")
    code = task_str + "\n" + impl_str
    with open(temp_test_path, 'w') as file:
        file.write(code)
    result = pytest.main([temp_test_path], plugins=[])
    os.remove(temp_test_path) # Comment this line to keep the test file when debugging
    return (result, impl_str, temp["usage"])

class QueryPyBody(Agent):
    def __init__(self, key, config: Storage):
        super().__init__(key, config)

    def run(self, storage: Storage):
        config_data = self.config.retrieve(self.key)
        num_samples = config_data["num_samples"]
        num_workers = config_data["num_workers"]
        storage_data = storage.retrieve(self.key)
        task_str = storage_data["task"]        
        prompt = storage_data["prompt"]
        success = []
        failure = []
        usages = []
        start = time.time()
        with tqdm(total=num_samples, smoothing=0) as pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        single_query_py_body,
                        id,
                        prompt,
                        task_str,
                        config_data
                    )
                    for id in range(num_samples)
                }
                for future in as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            if result[0] == pytest.ExitCode.OK:
                                success.append(result[1])
                            else:
                                failure.append(result[1])
                            usages.append(result[2])
                    except Exception as e:
                        print("Got an error!", e)
                        continue
        end = time.time()
        print_elapsed_time(start, end)
        storage_data["success"] = success
        storage_data["failure"] = failure # Stores a list of failed impls (not task+impl)
        storage_data["usage"] = usages
    
    def deduplicate_failure(self, storage: Storage):
        storage_data = storage.retrieve(self.key)
        failure = storage_data["failure"]
        task_str = storage_data["task"]
        code_history = ast_analysis.EquivalentSet(ast_analysis.check_program_equivalence)
        rep_list = []
        rep_impl_list = []
        for (id, impl_str) in enumerate(failure):
            code = task_str + "\n" + impl_str
            code_history.add(code, id)
        for (_, v) in code_history.item_map.items():
            rep_list.append(failure[v[0]])
            rep_impl_list.append(failure[v[0]])
        storage_data["failure_rep"] = rep_list
        storage_data["failure_rep_impl"] = rep_impl_list

    def deduplicate_success(self, storage: Storage):
        storage_data = storage.retrieve(self.key)
        failure = storage_data["success"]
        task_str = storage_data["task"]
        code_history = ast_analysis.EquivalentSet(ast_analysis.check_program_equivalence)
        rep_list = []
        rep_impl_list = []
        for (id, impl_str) in enumerate(failure):
            code = task_str + "\n" + impl_str
            code_history.add(code, id)
        for (_, v) in code_history.item_map.items():
            rep_list.append(failure[v[0]])
            rep_impl_list.append(failure[v[0]])
        storage_data["success_rep"] = rep_list
        storage_data["success_rep_impl"] = rep_impl_list

def single_query_affine_py_body(id, prompt, task_str, config_data):
    temp_dir = config_data["temp_dir"]
    model_name = config_data["model_name"]
    temperature = config_data["temperature"]
    max_tokens = config_data["max_tokens"]
    system_prompt = config_data["system_prompt"]
    temp = {"usage": {}}
    response = query_text(prompt, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens, storage=temp)
    impl = extract_code(response)
    impl_str = clean_code(impl)
    temp_test_path = os.path.join(temp_dir, f"test_{id}_{model_name}.py")
    code = task_str + "\n" + impl_str
    with open(temp_test_path, 'w') as file:
        file.write(code)
    result = pytest.main([temp_test_path], plugins=[])
    usage_all_once = count_usage.check_affine_type(temp_test_path)
    passed = usage_all_once and result == pytest.ExitCode.OK
    os.remove(temp_test_path) # Comment this line to keep the test file when debugging
    return (passed, impl_str, temp["usage"])

class QueryAffinePyBody(QueryPyBody):
    def __init__(self, key, config: Storage):
        super().__init__(key, config)

    def run(self, storage: Storage):
        config_data = self.config.retrieve(self.key)
        num_samples = config_data["num_samples"]
        num_workers = config_data["num_workers"]
        storage_data = storage.retrieve(self.key)
        task_str = storage_data["task"]        
        prompt = storage_data["prompt"]
        success = []
        failure = []
        usages = []
        start = time.time()
        with tqdm(total=num_samples, smoothing=0) as pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        single_query_affine_py_body,
                        id,
                        prompt,
                        task_str,
                        config_data
                    )
                    for id in range(num_samples)
                }
                for future in as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            if result[0]:
                                success.append(result[1])
                            else:
                                failure.append(result[1])
                            usages.append(result[2])
                    except Exception as e:
                        print("Got an error!", e)
                        continue
        end = time.time()
        print_elapsed_time(start, end)
        storage_data["success"] = success
        storage_data["failure"] = failure # Stores a list of failed impls (not task+impl)
        storage_data["usage"] = usages

def prepare_prompt_query_affine_rewrite(key, config: Storage, storage: Storage):
    config_data = config.retrieve(key)
    storage_data = storage.retrieve(key)
    example_path = config_data["example_path"]
    with open(example_path, "r") as f:
        example = f.read()
        storage_data["example"] = example
    test_str = storage_data["test"]
    prompt = f"""
```
{example}
```
Based on the above instruction, please add Copy and adjust the stream variables for the `impl` function.
```
{test_str}
```
Please output the code in this format:   
```
impl: |-
``` 
"""     
    storage_data["prompt"] = prompt
    return prompt

def single_query_affine_impl(id, prompt, task_data, config_data):
    temp_dir = config_data["temp_dir"]
    model_name = config_data["model_name"]
    temperature = config_data["temperature"]
    max_tokens = config_data["max_tokens"]
    system_prompt = config_data["system_prompt"]
    temp = {"usage": {}}
    response = query_text(prompt, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens, storage=temp)
    impl = extract_code(response)
    impl = clean_yaml(impl)
    temp_test_path = os.path.join(temp_dir, f"test_{id}_{model_name}.py")
    impl_data = yaml.safe_load(impl)
    if impl_data == None:
        raise Exception("Response Format Error!")
    data = {**task_data, **impl_data}
    code = yaml_to_code.yaml_to_code(data)
    with open(temp_test_path, 'w') as file:
        file.write(code)
    result = pytest.main([temp_test_path], plugins=[])
    usage_all_once = count_usage.check_affine_type(temp_test_path)
    passed = usage_all_once and result == pytest.ExitCode.OK
    os.remove(temp_test_path) # Comment this line to keep the test file when debugging
    return (passed, impl_data, temp["usage"])

class QueryAffineImpl(QueryImpl):
    def __init__(self, key, config: Storage):
        super().__init__(key, config)
    
    def run(self, storage: Storage):
        config_data = self.config.retrieve(self.key)
        num_samples = config_data["num_samples"]
        num_workers = config_data["num_workers"]
        storage_data = storage.retrieve(self.key)
        task_data = storage_data["task"]        
        prompt = storage_data["prompt"] # self.prepare_prompt(storage)
        success = []
        failure = []
        usages = []
        start = time.time()
        with tqdm(total=num_samples, smoothing=0) as pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        single_query_affine_impl,
                        id,
                        prompt,
                        task_data,
                        config_data
                    )
                    for id in range(num_samples)
                }
                for future in as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            if result[0]:
                                success.append(result[1])
                            else:
                                failure.append(result[1])
                            usages.append(result[2])
                    except Exception as e:
                        print("Got an error at QueryAffineImpl!", e)
                        continue
        end = time.time()
        print_elapsed_time(start, end)
        storage_data["success"] = success
        storage_data["failure"] = failure # Stores a list of failed impls (not task+impl)
        storage_data["usage"] = usages
    

def prepare_prompt_affine_impl_stage_0(key, config: Storage, storage: Storage):
    config_data = config.retrieve(key)
    storage_data = storage.retrieve(key)
    example_0_path = config_data["example_0_path"]
    with open(example_0_path, "r") as f:
        example_0 = f.read()
    task_data = storage_data["task"]
    task = yaml.dump(yaml_to_code.convert_to_literal(task_data), default_flow_style=False, sort_keys=False, width=float("inf"))
    prompt = f"""
```
{example_0}
```
Given the above example, complete the `impl` for the test
```
{task}
```
Please output the code in this format:   
```
impl: |-
``` 
"""     
    storage_data["prompt_0"] = prompt
    return prompt

def prepare_prompt_affine_impl_stage_1(example, test_str):
    prompt = f"""
```
{example}
```
Based on the above instruction, please add Copy and adjust the stream variables for the `impl` function if necessary.
```
{test_str}
```
Please output the code in this format:   
```
impl: |-
``` 
"""
    return prompt

def single_query_affine_impl_two_stages(id, prompt_0, example_1, task_data, config_data):
    temp_dir = config_data["temp_dir"]
    model_name = config_data["model_name"]
    temperature = config_data["temperature"]
    max_tokens = config_data["max_tokens"]
    system_prompt = config_data["system_prompt"]
    temp_0 = {"usage": {}}
    response = query_text(prompt_0, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens, storage=temp_0)
    stage_0_impl = extract_code(response)
    stage_0_impl_str = clean_yaml(stage_0_impl)
    prompt_1 = prepare_prompt_affine_impl_stage_1(example_1, stage_0_impl_str)
    temp_1 = {"usage": {}}
    response = query_text(prompt_1, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens, storage=temp_1)
    impl = extract_code(response)
    impl = clean_yaml(impl)
    temp_test_path = os.path.join(temp_dir, f"test_{id}_{yaml_to_code.clean_model_name(model_name)}.py")
    impl_data = yaml.safe_load(impl)
    data = {**task_data, **impl_data}
    code = yaml_to_code.yaml_to_code(data)
    with open(temp_test_path, 'w') as file:
        file.write(code)
    result = pytest.main([temp_test_path], plugins=[])
    usage_all_once = count_usage.check_affine_type(temp_test_path)
    passed = usage_all_once and result == pytest.ExitCode.OK
    os.remove(temp_test_path) # Comment this line to keep the test file when debugging
    return (passed, prompt_1, impl_data, {"stage0" : temp_0["usage"], "stage1" : temp_1["usage"]})


class QueryAffineImplTwoStages(QueryImpl):
    def __init__(self, key, config: Storage):
        super().__init__(key, config)
    
    def run(self, storage: Storage):
        config_data = self.config.retrieve(self.key)
        num_samples = config_data["num_samples"]
        num_workers = config_data["num_workers"]
        storage_data = storage.retrieve(self.key)
        task_data = storage_data["task"]        
        prompt_0 = storage_data["prompt_0"]
        example_1_path = config_data["example_1_path"]
        with open(example_1_path, "r") as f:
            example_1 = f.read()
        success_impl = []
        success_prompt = []
        failure_impl = []
        failure_prompt = []
        usages = []
        start = time.time()
        with tqdm(total=num_samples, smoothing=0) as pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        single_query_affine_impl_two_stages,
                        id,
                        prompt_0,
                        example_1,
                        task_data,
                        config_data
                    )
                    for id in range(num_samples)
                }
                for future in as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            if result[0]:
                                success_prompt.append(result[1])
                                success_impl.append(result[2])
                            else:
                                failure_prompt.append(result[1])
                                failure_impl.append(result[2])
                            usages.append(result[3])
                    except Exception as e:
                        print("Got an error!", e)
                        continue
        end = time.time()
        print_elapsed_time(start, end)
        storage_data["success"] = success_impl
        storage_data["success_prompt_1"] = success_prompt
        storage_data["failure"] = failure_impl # Stores a list of failed impls (not task+impl)
        storage_data["failure_prompt_1"] = failure_prompt
        storage_data["usage"] = usages
    
