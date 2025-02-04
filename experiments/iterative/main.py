import pandas as pd
import yaml
from tools import yaml_to_code, merge_best
import os
import glob
import csv
import argparse
from reinforce.base import Storage
from reinforce.actions import query_affine_impl_two_stages
import json
import numpy as np

def parallel_affine_impl_two_stage(model_name, base_path, example_0_path, example_1_path, task_path, exp_name, num_samples, temperature=0.7):
    temp_base_dir = os.path.join(base_path, exp_name)
    if not os.path.exists(temp_base_dir):
        os.makedirs(temp_base_dir)
    config = Storage()
    storage = Storage()
    key = "query_affine_impl_two_stage"
    query_affine_impl_two_stages(key, {
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": 1024,
        "system_prompt": "You are an AI assistant tasked with reasoning and generating code.",
        "num_samples": num_samples,
        "num_workers": min(num_samples, 40),
        "temp_base_dir": temp_base_dir,
        "example_0_path": example_0_path,
        "example_1_path": example_1_path,
        "task_path": task_path
    }, config, storage)
    # Store the usage
    temp_dir = config.retrieve(key)["temp_dir"]
    usage = storage.retrieve(key)["usage"]
    with open(os.path.join(temp_dir, "usage.json"), "w") as f:
        json.dump(usage, f, indent=4)
    stats = {}
    stats['success'] = len(storage.retrieve(key)['success'])
    stats['success_rep'] = len(storage.retrieve(key)['success_rep'])
    stats['failure'] = len(storage.retrieve(key)['failure'])
    stats['failure_rep'] = len(storage.retrieve(key)['failure_rep'])
    with open(os.path.join(temp_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    return (temp_dir, stats)

def call_agent(args):
    rows = []
    with open(args.input_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    for row in rows:
        exp_type = row['Type']
        task_path = row['Task']
        exp_name = exp_type + "/" + task_path.split('/')[-1].split('.')[0].replace('test', f'{args.num_samples}')
        temp_dir, stats = parallel_affine_impl_two_stage(args.model_name, args.base_path, args.example_0_path, args.example_1_path, task_path, exp_name, args.num_samples, args.temperature)
        row['ExpPath'] = temp_dir
        for (k, v) in stats.items():
            row[k] = v
        row['Trials'] = args.num_samples
        row['Temperature'] = args.temperature

    
    with open(args.output_csv, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def filter_failed(input_file, output_file):
    """
    Construct the test set
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Filter rows where success >= min_success
    filtered_df = df[(df['success'] == 0)]
    
    # Print summary
    print(f"Original number of rows: {len(df)}")
    print(f"Number of rows after filtering (failed): {len(filtered_df)}")
    
    # Only keep the ["Type", "Task", "Affine"] fields
    filtered_df = filtered_df[["Type", "Task", "Affine"]]
    # Save or print results

    filtered_df.to_csv(output_file, index=False)

def filter_hard(input_file, output_file, m):
    """
    Construct the training set
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Filter rows where success >= min_success
    filtered_df = df[(df['success'] > 0) & (df['success'] <= m)]
    
    # Print summary
    print(f"Original number of rows: {len(df)}")
    print(f"Number of rows after filtering (hard): {len(filtered_df)}")

    filtered_df.to_csv(output_file, index=False)

def pick_shortest_example(impl_data_dict_list):
    for impl in impl_data_dict_list:
        impl_str = impl["impl"]
        impl_str = yaml_to_code.clean_python_code(impl_str)
        num_lines = len(impl_str.split("\n"))
        impl["num_lines"] = num_lines
    sorted_impl_data_dict_list = sorted(impl_data_dict_list, key=lambda x: x["num_lines"])
    shortest_impl = sorted_impl_data_dict_list[0]
    del shortest_impl["num_lines"]
    return shortest_impl

def construct_base_prompt(original_file, hard_file, output_file):
    with open(original_file, "r") as f:
        original_prompt_str = f.read()
    example_dict = {"examples": []}

    # Read the CSV file
    with open(hard_file, "r") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            task_path = row["Task"]
            task_data = yaml.safe_load(open(task_path, "r"))
            exp_path = row["ExpPath"]
            
            # Get all success implementation YAML files
            impl_data_dict_list = []
            yaml_pattern = os.path.join(exp_path, "success_impl_rep_*.yaml")
            for yaml_file in glob.glob(yaml_pattern):
                with open(yaml_file, "r") as yf:
                    impl_data = yaml.safe_load(yf)
                    impl_data_dict_list.append(impl_data)
            
            if impl_data_dict_list:
                # Get the shortest example
                shortest_impl = pick_shortest_example(impl_data_dict_list)
                
                # Combine task data with implementation data
                data = {**task_data, **shortest_impl}
                del data["test_name"]
                example_dict["examples"].append(data)
            else:
                print(f"No success implementation YAML files found in {exp_path}")

    # Convert to YAML string with literal style
    example_str = yaml.dump(
        yaml_to_code.convert_to_literal(example_dict),
        default_flow_style=False,
        sort_keys=False,
        width=float("inf")
    )

    # Combine original prompt with examples
    output_prompt_str = original_prompt_str + "\n" +example_str
    
    # Write to output file
    with open(output_file, "w") as f:
        f.write(output_prompt_str)

def get_total_pass(result_file):
    return (pd.read_csv(result_file)['success'] > 0).sum()

def get_group_ms(result_file, num_groups):
    df = pd.read_csv(result_file)
    
    # Filter and sort successes greater than 0
    successes = sorted(df[df['success'] > 0]['success'].values)
    if len(successes) == 0:
        raise ValueError("No successes found in the result file.")
    ids = np.array_split(np.arange(len(successes)), num_groups)
    ms = [successes[id[-1]] for id in ids]
    return ms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_nickname", type=str, required=True)
    parser.add_argument("--example_1_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--base_date", type=str, required=True)
    parser.add_argument("--num_groups", type=int, required=True)
    parser.add_argument("--start_iter", type=int, default=0)
    args = parser.parse_args()

    project_path = os.getenv("STEPBASE", None)
    if project_path is None:
        raise ValueError("STEPBASE environment variable is not set.")
    base_date = args.base_date
    num_groups = args.num_groups
    ref_file = f"{project_path}/experiments/benchcard_fullpath.csv"
    iter = args.start_iter
    while True:
        ms = get_group_ms(f"{project_path}/experiments/{base_date}/result_{args.model_nickname}_merged_{iter}.csv", num_groups)
        print(ms)
        for m in ms:
            last_iter_result = f"{project_path}/experiments/{base_date}/result_{args.model_nickname}_merged_{iter}.csv"
            last_iter_failed = f"{project_path}/experiments/{base_date}/failed_tasks_{iter}.csv"
            last_iter_hard = f"{project_path}/experiments/{base_date}/hard_tasks_{iter}.csv"
            original_file = f"{project_path}/prompts/proposer_base.yaml"
            prompt_with_examples_file = f"{project_path}/experiments/{base_date}/step_with_examples_{iter}.yaml"
            filter_failed(last_iter_result, last_iter_failed)
            filter_hard(last_iter_result, last_iter_hard, m)
            construct_base_prompt(original_file, last_iter_hard, prompt_with_examples_file)

            args.input_csv = last_iter_failed
            args.example_0_path = prompt_with_examples_file
            result_files_temperatures = []
            for (temperature, temperature_str) in zip([0.4, 0.7, 1.0], ["04", "07", "10"]):
                args.output_csv = f"{project_path}/experiments/{base_date}/result_with_examples_{iter}_{temperature_str}.csv"
                result_files_temperatures.append(args.output_csv)
                args.temperature = temperature
                call_agent(args)
            result_file_merged = f"{project_path}/experiments/{base_date}/result_with_examples_merged_{iter}.csv"
            merge_best.keep_best_success(
                result_files_temperatures,
                result_file_merged, 
                ref_file
            )
            cur_iter_result = f"{project_path}/experiments/{base_date}/result_{args.model_nickname}_merged_{iter+1}.csv"
            result_files_to_merge = [last_iter_result, result_file_merged]
            merge_best.keep_best_success(
                result_files_to_merge,
                cur_iter_result,
                ref_file
            )
            
            last_iter_passed = get_total_pass(last_iter_result)
            cur_iter_passed = get_total_pass(cur_iter_result)
            iter += 1
            if cur_iter_passed > last_iter_passed:
                break
        if m == ms[-1] and cur_iter_passed == last_iter_passed:
            break


