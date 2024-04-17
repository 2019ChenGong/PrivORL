import numpy as np
import os


original_path = 'curiosity_driven_results_maze2d-umaze-dense-v1_0.3'
sample_name = 'maze2d-umaze-dense-v1_samples_1000000.0_10dp_0.5.npz'
data = np.load(os.path.join(original_path, sample_name))


for key in data.keys():
    array = data[key]
    print(f"Checking {key}:")
    if np.any(np.isnan(array)) or np.any(np.isinf(array)):
        print(" - Found NaN or Inf values.")
    if np.any(np.isnan(array)):
        print(" - Found missing values.")


def check_for_errors(data):
    data_types = ['observations', 'actions', 'next_observations', 'rewards', 'terminals']
    
    for i in range(len(data['observations'])):
        if i % 100 == 0:
            print(f"checked sample {i}")
        for data_type in data_types:
            array = data[data_type][i]

            if np.any(np.isnan(array)) or np.any(np.isinf(array)):
                print(f"Error found in sample {i}, type: {data_type}")
                print(f"Corresponding array: {array}")  

            if np.any(np.isnan(array)):
                print(f"Missing value found in sample {i}, type: {data_type}")


def remove_errors(data):
    data_types = ['observations', 'actions', 'next_observations', 'rewards', 'terminals']
    cleaned_data = {key: [] for key in data_types}
    valid_indices = []
    check_batch = 100  # Number of samples to process at once
    
    for i in range(0, len(data['observations']), check_batch):
    # for i in range(0, 50, batch_size):
        if i % check_batch == 0:
            print(f'successfully checked sample {i}')
        batch_valid_indices = []  # Store valid indices for this batch
        end_idx = min(i + check_batch, len(data['observations']))

        obs = data['observations'][i:end_idx]
        actions = data['observations'][i:end_idx]        
        next_obs = data['next_observations'][i:end_idx]
        rewards = data['rewards'][i:end_idx]
        terminals = data['terminals'][i:end_idx].astype(np.float32)
        array = np.concatenate([obs, actions, rewards[:, None], next_obs, terminals[:, None]], axis=1)

        # Check for nan and inf in the concatenated data
        if not np.any(np.isnan(array)) and not np.any(np.isinf(array)):
            batch_valid_indices = list(range(i, end_idx))
            # Append the entire batch of valid samples into cleaned_data
            for data_type in data_types:
                cleaned_data[data_type].extend(data[data_type][i:end_idx])
                # print(f'saved samples {i} to {end_idx}')
        else:
            print('there are issue samples')
            # If issues found, check each sample individually
            for j in range(i, end_idx):
                concatenated_data_single = np.concatenate([data[data_type][j].flatten() for data_type in data_types])
                if not np.any(np.isnan(concatenated_data_single)) and not np.any(np.isinf(concatenated_data_single)):
                    batch_valid_indices.append(j)
                    print(f'successfully checked sample {j}')
                    for data_type in data_types:
                        cleaned_data[data_type].append(data[data_type][j])
        
        # Add valid indices to the overall list
        valid_indices.extend(batch_valid_indices)
        
    cleaned_data = {key: np.array(cleaned_data[key]) for key in cleaned_data}

    # Save cleaned data
    np.savez(os.path.join(original_path, f'cleaned_{sample_name}'), **cleaned_data)
    
    return valid_indices



valid_indices = remove_errors(data)
print(f"Removed {len(data['observations']) - len(valid_indices)} samples with errors.")

