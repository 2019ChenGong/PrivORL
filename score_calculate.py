import os
import json
import statistics

def calculate_and_update_average_scores(directory):
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir.startswith('pgm'):
                json_path = os.path.join(root, dir, 'eval_0.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                        scores = data.get('d4rl_normalized_score', [])
                        if len(scores) >= 10:
                            last_ten_scores = scores[-10:]
                            average_score = sum(last_ten_scores) / len(last_ten_scores)
                            std_deviation = statistics.stdev(last_ten_scores)
                            data['average_last_ten_scores'] = average_score
                            data['standard_deviation_last_ten_scores'] = std_deviation
                            with open(json_path, 'w') as file_to_write:
                                json.dump(data, file_to_write, indent=4)
                            print(f"Updated {json_path} with average of last ten scores: {average_score} and standard deviation: {std_deviation}")
                        else:
                            print(f"Not enough scores in {json_path}")
                else:
                    print(f"File not found: {json_path}")


calculate_and_update_average_scores('corl_logs_antmaze')



def rename_directories(directory):
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if 'pategan_eps_1' in dir_name:
                new_name = dir_name.replace('pategan_eps_1', 'pategan')
                os.rename(os.path.join(root, dir_name), os.path.join(root, new_name))
                print(f"Renamed directory: {dir_name} -> {new_name}")

# rename_directories('corl_logs_2mazed')