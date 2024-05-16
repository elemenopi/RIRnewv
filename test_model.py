from train.custom_dataset import customdataset
import train.network as network
import numpy as np
import torch
import multiprocessing
import torch.nn.functional as F
from scipy.io.wavfile import read,write
import utils
#utils.change_sampling_rate("C://Users//lipov//Documents//GitHub//project//RIRnewv//OUTPUTS//output016")
data_test = customdataset(input_dir = "C://Users//lipov//Documents//GitHub//project//RIRnewv//OUTPUTS",n_mics = 6,sr = 44100,window_idx=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = False and torch.cuda.is_available()
num_workers = 3
num_workers = min(multiprocessing.cpu_count(), num_workers)
kwargs = {
    'num_workers': num_workers,
    'pin_memory': True
} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=1,
                                              **kwargs)

model = network.CoSNetwork()
model.load_state_dict(torch.load('C:\\Users\\lipov\\Documents\\GitHub\\project\\RIRnewv\\synthetic_6mics_.0725m_44100kHz.pt', map_location='cpu'))
model.eval()

for batch_idx,(data,label_voice_signals,window_idx) in enumerate(test_loader):
    data = data.to(device)
    print("label voices signal")
    print(label_voice_signals.shape)
    print(label_voice_signals)
    for i in range(6):
        output_voice_np = data[0][i].detach().cpu().numpy()
        write(f"model_results\\datavoices_result_batch{batch_idx}_channel{i}.wav", 44100, output_voice_np)
        output_voice_np = label_voice_signals[0][i].detach().cpu().numpy()
        write(f"model_results\\labelvoices_result_batch{batch_idx}_channel{i}.wav", 44100, output_voice_np)
    label_voice_signals = label_voice_signals.to(device)
    window_idx = window_idx.to(device)

    data, means, stds = network.normalize_input(data)
    valid_length = model.valid_length(data.shape[-1])
    delta = valid_length - data.shape[-1]
    padded = F.pad(data, (delta // 2, delta - delta // 2))
    output_signal = model(padded, window_idx)
    output_signal = network.center_trim(output_signal, data)
    output_signal = network.unnormalize_input(output_signal, means, stds)
    output_voices = output_signal[:,0]
    for i in range(6):
        output_voice_np = output_voices[0][i].detach().cpu().numpy()
        write(f"model_results\\model_result_batch{batch_idx}_channel{i}.wav", 44100, output_voice_np)