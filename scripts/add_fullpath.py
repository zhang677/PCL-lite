import os
import csv

base_path = os.getenv("STEPBASE", None)
if base_path is None:
    raise ValueError("STEPBASE environment variable is not set.")
bench_base_path = os.path.join(base_path, "benchmark")
exp_base_path = os.path.join(base_path, "experiments")
# 1. Read benchcard.csv
# 2. For each row, change "Task" to os.path.join(bench_base_path, row["Task"])
# 3. Write to a new file called benchcard_fullpath.csv
with open(os.path.join(exp_base_path, "benchcard.csv"), 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    for row in rows:
        row['Task'] = os.path.join(bench_base_path, row['Task'])
    with open(os.path.join(exp_base_path, "benchcard_fullpath.csv"), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

