# import numpy as np
# import os
# import concurrent.futures

# # for key in data.keys():
# #     array = data[key]
# #     print(f"Checking {key}:")
# #     if np.any(np.isnan(array)) or np.any(np.isinf(array)):
# #         print(" - Found NaN or Inf values.")
# #     if np.any(np.isnan(array)):
# #         print(" - Found missing values.")


# def check_for_errors(data):
#     data_types = ['observations', 'actions', 'next_observations', 'rewards', 'terminals']
    
#     for i in range(len(data['observations'])):
#         if i % 100 == 0:
#             print(f"checked sample {i}")
#         for data_type in data_types:
#             array = data[data_type][i]

#             if np.any(np.isnan(array)) or np.any(np.isinf(array)):
#                 print(f"Error found in sample {i}, type: {data_type}")
#                 print(f"Corresponding array: {array}")  

#             if np.any(np.isnan(array)):
#                 print(f"Missing value found in sample {i}, type: {data_type}")


# def remove_errors(original_path, sample_name):
#     data = np.load(os.path.join(original_path, sample_name))
#     data_types = ['observations', 'actions', 'next_observations', 'rewards', 'terminals']
#     cleaned_data = {key: [] for key in data_types}
#     valid_indices = []
#     check_batch = 5000  # number of samples to process at once
    
#     for i in range(0, len(data['observations']), check_batch):
#         if i % check_batch == 0:
#             print(f'successfully checked sample {i}')
#         batch_valid_indices = []  # store valid indices for this batch
#         end_idx = min(i + check_batch, len(data['observations']))

#         obs = data['observations'][i:end_idx]
#         actions = data['observations'][i:end_idx]        
#         next_obs = data['next_observations'][i:end_idx]
#         rewards = data['rewards'][i:end_idx]
#         terminals = data['terminals'][i:end_idx].astype(np.float32)
#         array = np.concatenate([obs, actions, rewards[:, None], next_obs, terminals[:, None]], axis=1)

#         # check for nan and inf in the concatenated data
#         if not np.any(np.isnan(array)) and not np.any(np.isinf(array)):
#             batch_valid_indices = list(range(i, end_idx))
#             # Append the entire batch of valid samples into cleaned_data
#             for data_type in data_types:
#                 cleaned_data[data_type].extend(data[data_type][i:end_idx])
#                 # print(f'saved samples {i} to {end_idx}')
#         else:
#             print('there are issue samples')
#             # if issues found, check each sample individually
#             for j in range(i, end_idx):
#                 concatenated_data_single = np.concatenate([data[data_type][j].flatten() for data_type in data_types])
#                 if not np.any(np.isnan(concatenated_data_single)) and not np.any(np.isinf(concatenated_data_single)):
#                     batch_valid_indices.append(j)
#                     print(f'successfully checked sample {j}')
#                     for data_type in data_types:
#                         cleaned_data[data_type].append(data[data_type][j])
        
#         # add valid indices to the overall list
#         valid_indices.extend(batch_valid_indices)
        
#     cleaned_data = {key: np.array(cleaned_data[key]) for key in cleaned_data}

#     # save cleaned data
#     np.savez(os.path.join(original_path, f'cleaned_{sample_name}'), **cleaned_data)
    
#     print(f"Removed {len(data['observations']) - len(valid_indices)} samples with errors.")

import torch
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

def load_npz_to_torch(path, device='cuda'):
    npz_file = np.load(path)
    data = {k: torch.tensor(npz_file[k], device=device, dtype=torch.float32) for k in npz_file.files}
    return data

def check_and_clean_data(data, start_idx, end_idx):
    data_types = ['observations', 'actions', 'next_observations', 'rewards', 'terminals']
    cleaned_data = {key: [] for key in data_types}
    valid_indices = []
    for i in range(start_idx, end_idx):
        valid = True
        for data_type in data_types:
            array = data[data_type][i]
            if torch.any(torch.isnan(array)) or torch.any(torch.isinf(array)):
                valid = False
                break
        if valid:
            valid_indices.append(i)
            for data_type in data_types:
                cleaned_data[data_type].append(data[data_type][i].cpu().numpy())
    return cleaned_data, valid_indices

def process_data_chunk(path, device, start_idx, end_idx):
    data = load_npz_to_torch(path, device)
    cleaned_data, valid_indices = check_and_clean_data(data, start_idx, end_idx)
    return cleaned_data, valid_indices

def merge_data(cleaned_data_list):
    merged_data = {key: [] for key in cleaned_data_list[0].keys()}
    for data in cleaned_data_list:
        for key in merged_data.keys():
            merged_data[key].extend(data[key])
    return merged_data

def save_cleaned_data(original_path, sample_name, cleaned_data):
    for key in cleaned_data.keys():
        cleaned_data[key] = np.array(cleaned_data[key])
    np.savez(os.path.join(original_path, sample_name), **cleaned_data)

def remove_errors(original_path, sample_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = os.path.join(original_path, sample_name)
    data = load_npz_to_torch(path, device)
    total_samples = len(data['observations'])
    chunk_size = 5000

    cleaned_data_list = []
    valid_indices_list = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, total_samples, chunk_size):
            end_idx = min(i + chunk_size, total_samples)
            futures.append(executor.submit(process_data_chunk, path, device, i, end_idx))

        
        count = 0
        for future in futures:
            cleaned_data, valid_indices = future.result()
            cleaned_data_list.append(cleaned_data)
            valid_indices_list.extend(valid_indices)
            count = count + 1
            print(count)

    merged_data = merge_data(cleaned_data_list)

    save_cleaned_data(original_path, sample_name, merged_data)
    print(f"Cleaned data saved as {sample_name}")
    print(f"Removed {total_samples - len(valid_indices_list)} samples with errors.")




if __name__ == '__main__':
    # datasets = ['antmaze-umaze-v1', 'antmaze-medium-play-v1', 'antmaze-large-play-v1']
    # datasets = ["maze2d-umaze-dense-v1", "maze2d-medium-dense-v1", "maze2d-large-dense-v1"]
    # datasets = ["maze2d-medium-dense-v1"]
    
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     for dataset in datasets:
    #         # original_path = f'curiosity_driven_results_{dataset}_0.3'
    #         original_path = f'curiosity_driven_results_{dataset}_0.3'
    #         sample_name = f'{dataset}_samples_1000000.0_10dp_0.5.npz'
    #         executor.submit(remove_errors, original_path, sample_name)

    # for dataset in datasets:
    #         # original_path = f'curiosity_driven_results_{dataset}_0.3'
    #         original_path = f'results_{dataset}_0.3'
    #         sample_name = f'{dataset}_samples_1000000.0_10dp_0.5.npz'
    #         remove_errors(original_path, sample_name)
    dataset = "maze2d-medium-dense-v1"
    

    # original_path = f'curiosity_driven_results_{dataset}_0.3'
    original_path = f'curiosity_driven_results_{dataset}_0.3'
    sample_name = f'{dataset}_samples_1000000.0_10dp_0.5.npz'
    remove_errors(original_path, sample_name)


