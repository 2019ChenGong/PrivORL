import os
import json
import numpy as np

def find_json_files(root_dir):
    target_files = []
    for root, dirs, files in os.walk(root_dir):
        if "medium" in root:
            if '0.4CurRate' in root:
                if 'epsilon_20' in root:
                    for file in files:
                        if file == "eval_0.json":
                            target_files.append(os.path.join(root, file))
    return target_files

def find_mariginal_json_files(root_dir, cur_rate, epsilon):
    target_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if "umaze" in file:
                if f'epsilon_{epsilon}' in file:
                    if f'{cur_rate}CurRate' in file:
                        for file in files:
                            if file == f"{cur_rate}CurRate_maze2d-umaze-dense-v1_epsilon_{epsilon}_quality_report.json":
                                target_files.append(os.path.join(root, file))
    return target_files

def compute_averages(json_files):
    average_scores = []
    std_deviations = []

    for file_path in json_files:
        with open(file_path, 'r') as file:
            data = json.load(file)
            average_scores.append(data["average_last_ten_scores"])
            std_deviations.append(data["standard_deviation_last_ten_scores"])

    return np.mean(average_scores), np.mean(std_deviations)


def compute_marginal_averages(json_files):
    marginals = []
    cors = []

    for file_path in json_files:
        with open(file_path, 'r') as file:
            data = json.load(file)
            print(data)
            marginals.append(data["Column Shapes"])
            cors.append(data["Column Pair Trends"])

    return np.mean(marginals), np.mean(cors)

def d4rl():
    root_directory = 'corl_logs_param_analysis_maze2d'
    # root_directory = 'corl_logs_ablation_walker2d'

    json_files = find_json_files(root_directory)

    average_last_ten_scores, average_std_deviation = compute_averages(json_files)

    print(f"Average of 'average_last_ten_scores': {average_last_ten_scores:.6f}")
    print(f"Average of 'standard_deviation_last_ten_scores': {average_std_deviation:.6f}")

def marginal():
    root_directory = 'marginal_results_param_analysis'

    cur_rate = 0.5
    epsilon = 15

    json_files = find_mariginal_json_files(root_directory, cur_rate, epsilon)
    print(json_files)

    average_marginal, average_cor = compute_marginal_averages(json_files)

    print(f"Average of 'marginal': {average_marginal:.6f}")
    print(f"Average of 'cor': {average_cor:.6f}")

# marginal()

d4rl()