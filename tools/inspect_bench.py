import csv
from pathlib import Path
import json

def calculate_mean_output_tokens(path, mode="agent"):
    json_file = Path(path) / "usage.json"
    try:
        # Read and parse the JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)
        # Extract output_tokens values
        if mode == "agent":
            if 'sonnet' in path:
                output_tokens =  [entry['stage0']['output_tokens'] + entry['stage1']['output_tokens'] for entry in data]
            else:
                output_tokens = [entry['stage0']['completion_tokens'] + entry['stage1']['completion_tokens'] for entry in data]
        elif mode == "single":
            if 'sonnet' in path:
                output_tokens = [entry['output_tokens'] for entry in data]
            else:
                output_tokens = [entry['completion_tokens'] for entry in data]
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # Calculate mean
        mean_output = sum(output_tokens) / len(output_tokens)
        # Extract input_tokens values
        if mode == "agent":
            if 'sonnet' in path:
                input_tokens =  [entry['stage0']['input_tokens'] for entry in data]
            else:
                input_tokens = [entry['stage0']['prompt_tokens'] for entry in data]
        elif mode == "single":
            if 'sonnet' in path:
                input_tokens = [entry['input_tokens'] for entry in data]
            else:
                input_tokens = [entry['prompt_tokens'] for entry in data]
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # Calculate mean
        mean_input = sum(input_tokens) / len(input_tokens)

        return int(mean_input), int(mean_output)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return 0, 0
    
def process_csv_get_token_counts(input_csv, output_csv):
    # Read the input CSV
    rows = []
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Add example_path for each row
    for row in rows:
        path = row['ExpPath']
        mean_input, mean_output = calculate_mean_output_tokens(path)
        row['InpuT'] = mean_input
        row['OutpuT'] = mean_output

    # Write the output CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)