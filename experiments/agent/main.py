import argparse
from reinforce.base import Storage
from reinforce.actions import query_affine_impl_two_stages
import json
import csv
import os

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--example_0_path", type=str, required=True)
    parser.add_argument("--example_1_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    args = parser.parse_args()
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