from custom_dataset import customdataset
import os
import tqdm
import torch
from torch.utils.data import DataLoader
import time

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'OUTPUTS'))
dataset = customdataset(input_dir=data_dir, window_idx=2)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Test the dataloader to see if it's slowing down
loading_times = []

for i, (mixed_data, target_voice_data, _) in enumerate(tqdm.tqdm(dataloader, total=10)):
    # Measure the time taken for each batch to be loaded
    start_time = time.time()

    # Simulate some processing (e.g., moving data to device)
    mixed_data = mixed_data.to('cpu')
    target_voice_data = target_voice_data.to('cpu')

    end_time = time.time()
    loading_time = (end_time - start_time) * 1000  # Time in milliseconds
    loading_times.append(loading_time)

    if i == 10:
        break

# Calculate the average loading time
average_loading_time = sum(loading_times) / len(loading_times)
print(f'Average loading time per batch: {average_loading_time:.2f} ms')
