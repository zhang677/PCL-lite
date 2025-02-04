import pandas as pd

def keep_best_success(input_files, output_file, ref_file):
    # Read all the files
    dfs = [pd.read_csv(file) for file in input_files]

    # Merge all the files
    df = pd.concat(dfs)

    # Keep the best success record
    df = df.sort_values(by=['success', 'ExpPath'], ascending=[False, False]).drop_duplicates(subset=['Task'], keep='first')
    
    # Sort the df based on the order of "Task" of ref_file
    ref_df = pd.read_csv(ref_file)
    task_order = {type_: idx for idx, type_ in enumerate(ref_df['Task'].unique())}
    df = df.sort_values(by=['Task'], key=lambda x: x.map(task_order))
    # Save the result
    df.to_csv(output_file, index=False)
