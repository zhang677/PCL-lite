import os
import time
import yaml
import json
import subprocess
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

def single_query_plan(id, prompt, config_data):
    temp_dir = config_data["temp_dir"]
    model_name = config_data["model_name"]
    temperature = config_data["temperature"]
    max_tokens = config_data["max_tokens"]
    system_prompt = config_data["system_prompt"]
    if "claude" in model_name:
        response = query_anthropic(prompt, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens)['text']
        subtasks = extract_code(response)
    else:
        response = query(prompt, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens)
        subtasks = extract_code(response)
    subtasks = clean_yaml(subtasks)
    data = yaml.safe_load(subtasks)
    subtasks = data.get("subtasks", [])
    task_data = []
    for i, subtask_ori in enumerate(subtasks):
        subtask = subtask_ori
        code = yaml_to_code.yaml_plan_to_code(subtask)
        temp_test_path = os.path.join(temp_dir, f"test_{id}_{i}_{model_name}.py") 
        with open(temp_test_path, 'w') as file:
            file.write(code)
        result = pytest.main([temp_test_path], plugins=[])
        os.remove(temp_test_path)
        task_data.append((result, subtask_ori, i)) # Stores the subtask step id. The intuition is that the model later should try the early steps first, and early steps are shape operations, which are generally harder.
    return task_data

class QueryPlan(Agent):
    def __init__(self, key, config: Storage):
            super().__init__(key, config)

    def prepare_prompt(self, storage: Storage):
        config_data = self.config.retrieve(self.key)
        storage_data = storage.retrieve(self.key)
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
Please decompose the data_transform required by the test into smaller subtasks. If you use fns, please follow the format in the example. Output the code in this format:   
```
subtasks:
- name:
  inputs:
  - name:
    dtype:
    dims:
    data_gen:
  ...
  outputs:
  - name:
    dtype:
    dims:
    data_transform:
  ...

``` 
The subtasks should be concise. The data_transform can only use combination of `unsqueeze`, `repeat`, and `sum`. 
You can also provide thoughts on how to decompose the tasks. However, please make sure that your answer ends with the format above.
"""
        storage_data["prompt"] = prompt
        return prompt

    def run(self, storage):
        config_data = self.config.retrieve(self.key)
        num_samples = config_data["num_samples"]
        prompt = self.prepare_prompt(storage)
        success = []
        failure = []
        start = time.time()
        with tqdm(total=num_samples, smoothing=0) as pbar:
            with ProcessPoolExecutor(max_workers=config_data["num_workers"]) as executor:
                futures = {
                    executor.submit(
                        single_query_plan,
                        id,
                        prompt,
                        config_data
                    )
                    for id in range(num_samples)
                }
                for future in as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            for (status, data, step) in result:
                                if status == pytest.ExitCode.OK:
                                    success.append((data, step))
                                else:
                                    failure.append((data, step))
                    except Exception as e:
                        print("Got an error!", e)
                        continue
        end = time.time()
        print_elapsed_time(start, end)
        storage_data = storage.retrieve(self.key)
        storage_data["success"] = success
        storage_data["failure"] = failure

    def top_kd_plan(self, storage: Storage):
        config_data = self.config.retrieve(self.key)
        storage_data = storage.retrieve(self.key)
        K = config_data["max_plan_width"]
        D = config_data["max_plan_depth"]
        yaml_data_step = storage_data["success"]
        # Get the longest plan
        max_step = 0
        for (data, step) in yaml_data_step:
            max_step = max(max_step, step)
        code_histories = []
        for _ in range(max_step + 1):
            code_histories.append(ast_analysis.EquivalentSet(ast_analysis.check_program_equivalence))

        for (id, (data, step)) in enumerate(yaml_data_step):
            code_text = yaml_to_code.yaml_plan_to_code(data)
            code_histories[step].add(code_text, id)
        ids_dict = {}
        subtask_list = []
        subtask_step_list = []
        for (step, code_history) in enumerate(code_histories):
            id_lists = []
            for (_, v) in code_history.item_map.items():
                id_lists.append(v)
            id_lists.sort(key=lambda x: len(x), reverse=True)
            ids_dict[step] = id_lists
            if step < D:
                for (id, id_list) in enumerate(id_lists[:K]):
                    subtask_step_list.append((step, id))
                    subtask_list.append(yaml_data_step[id_list[0]][0])
        print(max_step)
        print(ids_dict)
        storage_data["subtasks"] = subtask_list
        storage_data["subtask_steps"] = subtask_step_list
        storage_data["ids_dict"] = ids_dict

def single_query_value(prompt, config_data):
    model_name = config_data["model_name"]
    temperature = config_data["temperature"]
    max_tokens = config_data["max_tokens"]
    system_prompt = config_data["system_prompt"]
    if "claude" in model_name:
        response = query_anthropic(prompt, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens)['text']
        value = extract_code(response)
    else:
        response = query(prompt, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens)
        value = extract_code(response)
    value = clean_yaml(value)
    data = yaml.safe_load(value)
    return data

class QueryValue(Agent):
    def __init__(self, key, config: Storage):
            super().__init__(key, config)
    
    def collect_error(self, storage: Storage):
        config_data = self.config.retrieve(self.key)
        storage_data = storage.retrieve(self.key)
        temp_dir = config_data["temp_dir"]
        model_name = config_data["model_name"]
        error_impls = storage_data["error_impls"]
        error_messages = {}
        for (id, data) in enumerate(error_impls):
            code = yaml_to_code.yaml_to_code(data)
            temp_test_path = os.path.join(temp_dir, f"test_{id}_{model_name}.py")
            with open(temp_test_path, 'w') as file:
                file.write(code)
            result = subprocess.run(
                ["pytest", "-v", temp_test_path, "--capture=no", "--no-header"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True  # Automatically decode output
            )
            assert result.returncode != 0
            error_messages[id] = str(result)
            os.remove(temp_test_path)
        
        storage_data["error_msg"] = error_messages
    
    def prepare_prompt(self, storage: Storage):
        storage_data = storage.retrieve(self.key)
        error_messages = storage_data["error_msg"]
        error_messages_str = json.dumps(error_messages)
        prompt = f"""
{error_messages_str}

Please return a yaml where each task is the key, and value is:
1. Score that measures the distance to the correct answer ranging from 0 to 10, the higher the better. 
2. The reason it is wrong, and the instruction to fix it without naming any specific operator
The output format should be:
```
subtask_0:
  score: 
  error:
  fix:
...
```
You can also provide thoughts on these scores and reasons. However, please make sure that your answer ends with the format above.
"""
        storage_data["prompt"] = prompt
        return prompt

    def run(self, storage: Storage):
        # Don't know how to deal with mutiple value responses and multiple max scores.
        # Therefore, num_samples is always 1. However, returned feedback is a list.
        config_data = self.config.retrieve(self.key)
        self.collect_error(storage)
        prompt = self.prepare_prompt(storage)
        data = single_query_value(prompt, config_data)
        storage_data = storage.retrieve(self.key)
        max_score = 0
        for id in storage_data["error_msg"].keys():
            feedback = data.get(f"subtask_{id}", {})
            score = feedback.get("score", 0)
            if score > max_score:
                max_score = score

        max_scores_impls = []
        max_scores_errors = []
        max_scores_fixes = []
        error_impls = storage_data["error_impls"]
        for id in storage_data["error_msg"].keys():
            feedback = data.get(f"subtask_{id}", {})
            score = feedback.get("score", 0)
            if score == max_score:
                max_scores_impls.append(error_impls[id])
                max_scores_errors.append(feedback.get("error", ""))
                max_scores_fixes.append(feedback.get("fix", ""))
        
        storage_data["full_feedback"] = [data]
        storage_data["feedback"] = [
            {
                **impl,
                "judgement": False,
                "reason": error # + " " + fix
            }
            for (impl, error, fix) in zip(max_scores_impls, max_scores_errors, max_scores_fixes)
        ]


class QueryValueMaxLength(Agent):
    def __init__(self, key, config: Storage):
            super().__init__(key, config)
    
    def collect_error(self, storage: Storage):
        config_data = self.config.retrieve(self.key)
        storage_data = storage.retrieve(self.key)
        temp_dir = config_data["temp_dir"]
        model_name = config_data["model_name"]
        error_impls = storage_data["error_impls"]
        error_messages = {}
        for (id, data) in enumerate(error_impls):
            code = yaml_to_code.yaml_to_code(data)
            temp_test_path = os.path.join(temp_dir, f"test_{id}_{model_name}.py")
            with open(temp_test_path, 'w') as file:
                file.write(code)
            result = subprocess.run(
                ["pytest", "-v", temp_test_path, "--capture=no", "--no-header"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True  # Automatically decode output
            )
            assert result.returncode != 0
            error_messages[id] = str(result)
            os.remove(temp_test_path)
        
        storage_data["error_msg"] = error_messages
    
    def prepare_prompt(self, storage: Storage):
        storage_data = storage.retrieve(self.key)
        error_messages = storage_data["error_msg"]
        error_messages_str = json.dumps(error_messages)
        prompt = f"""
{error_messages_str}

Please return a yaml where each task is the key, and value is:
1. Score that measures the distance to the correct answer ranging from 0 to 10, the higher the better. 
2. The reason it is wrong, and the instruction to fix it without naming any specific operator
The output format should be:
```
subtask_0:
  score: 
  error:
  fix:
...
```
You can also provide thoughts on these scores and reasons. However, please make sure that your answer ends with the format above.
"""
        storage_data["prompt"] = prompt
        return prompt

    def run(self, storage: Storage):
        # Don't know how to deal with mutiple value responses and multiple max scores.
        # Therefore, num_samples is always 1. However, returned feedback is a list.
        config_data = self.config.retrieve(self.key)
        self.collect_error(storage)
        prompt = self.prepare_prompt(storage)
        data = single_query_value(prompt, config_data)
        storage_data = storage.retrieve(self.key)
        max_length = 0
        code_length = []
        for (id, error_impl) in enumerate(storage_data["error_impls"]):
            code_string = error_impl.get("impl", "")
            # Count the number of effective lines of code
            lines = code_string.split('\n')
    
            # Counter for non-comment lines
            count = 0
            
            for line in lines:
                # Remove inline comments
                line = line.split('#')[0].strip()
                
                # Count non-empty lines
                if line:
                    count += 1
            max_length = max(max_length, count)
            code_length.append(count)

        max_scores_impls = []
        max_scores_errors = []
        max_scores_fixes = []
        error_impls = storage_data["error_impls"]
        for id in storage_data["error_msg"].keys():
            feedback = data.get(f"subtask_{id}", {})
            if code_length[id] == max_length:
                max_scores_impls.append(error_impls[id])
                max_scores_errors.append(feedback.get("error", ""))
                max_scores_fixes.append(feedback.get("fix", ""))
        
        storage_data["full_feedback"] = [data]
        storage_data["feedback"] = [
            {
                **impl,
                "judgement": False,
                "reason": error + fix
            }
            for (impl, error, fix) in zip(max_scores_impls, max_scores_errors, max_scores_fixes)
        ]

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
    
