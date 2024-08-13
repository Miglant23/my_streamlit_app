import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
import streamlit.components.v1 as components
import os
import numpy as np


st.write("""
# Visualisation of the latent features of bender data
Latent features have been obtained from using a convolutional autoencoder on twin channel bender signals, input and output. The data was able to be compressed into 256 latent feautures
""")

linearPCA_plot_path = os.path.join(os.path.dirname(__file__), 'linearPCA_all_data.html')
try:
    with open(linearPCA_plot_path, 'r',encoding='utf-8') as f:
        linearPCA_html = f.read()
    components.html(linearPCA_html, height=500)
except FileNotFoundError:
    st.error(f"The file at {linearPCA_plot_path} was not found.")
except Exception as e:
    st.error(f"An error occurred: {e}")

nonlinearPCA_plot_path = os.path.join(os.path.dirname(__file__), 'nonlinearPCA_all_data.html')
try:
    with open(nonlinearPCA_plot_path, 'r',encoding='utf-8') as f:
        nonlinearPCA_html = f.read()
    components.html(nonlinearPCA_html, height=500)
except FileNotFoundError:
    st.error(f"The file at {nonlinearPCA_plot_path} was not found.")
except Exception as e:
    st.error(f"An error occurred: {e}")        

#umap_plot_path=r'C:\Users\nakon\Desktop\my_streamlit_app\umap.html'
umap_plot_path = os.path.join(os.path.dirname(__file__), 'umap_combined_red_points.html')
try:
    with open(umap_plot_path, 'r',encoding='utf-8') as f:
        umap_html = f.read()
    components.html(umap_html, height=500)
except FileNotFoundError:
    st.error(f"The file at {umap_plot_path} was not found.")
except Exception as e:
    st.error(f"An error occurred: {e}")

umap_plot_path = os.path.join(os.path.dirname(__file__), 'umap_combined_colours.html')
try:
    with open(umap_plot_path, 'r',encoding='utf-8') as f:
        umap_html = f.read()
    components.html(umap_html, height=500)
except FileNotFoundError:
    st.error(f"The file at {umap_plot_path} was not found.")
except Exception as e:
    st.error(f"An error occurred: {e}")

dataset_path= os.path.join(os.path.dirname(__file__), 'combined_csv_dataset.pth')

#@st.cache_data
def load_dataset(dataset_path):
    dataset=torch.load(dataset_path)
    hash_map = {dataset[i][-1].item(): dataset[i] for i in range(len(dataset))}
    return dataset,hash_map
dataset,hash_map=load_dataset(dataset_path)


# Autoencoder Architecture
class Conv1DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv1DAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),

            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=0),  # Adjusted padding to 0
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),

            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=0),  # Adjusted padding to 0
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
        )

        # Latent space compression
        self.flatten = nn.Flatten()
        self.latent = nn.Linear(512 * 92, 256)

        # Decoder
        self.decoder_fc = nn.Linear(256, 512 * 92)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=0, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),

            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=0, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),

            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=0, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),

            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),

            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),

            nn.ConvTranspose1d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.latent(x)
        x = self.decoder_fc(x)
        x = x.view(-1, 512, 92)  # Reshape to match the decoder input
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.latent(x)
        return x

model_path = os.path.join(os.path.dirname(__file__), 'MainAutoencoder_V4_256D.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Conv1DAutoencoder().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


current_dir = os.getcwd()
new_dataset_path = os.path.join(current_dir, 'dataset_new_NEWEST.pth')

@st.cache_data
def load_new_dataset(new_dataset_path):
    new_dataset_load = torch.load(new_dataset_path)
    new_signals=new_dataset_load['data']
    new_frequencies=new_dataset_load['frequencies']
    new_labels=new_dataset_load['labels']
    new_indices=new_dataset_load['indices']
    new_times=new_dataset_load['times']
    return new_signals,new_frequencies,new_labels,new_indices,new_times
new_signals,new_frequencies,new_labels,new_indices,new_times=load_new_dataset(new_dataset_path)


types=['Sine','Square','Triangular']
#Combined data plotter
plotting_index=st.number_input('Select the index of sample for plotting:',min_value=0, max_value=len(new_signals)+1575,step=1)
if plotting_index<1575: #Data is Wenzhang
    sample_data=hash_map.get(plotting_index)
    signals=sample_data[1]
    signals_tensor = torch.tensor(signals).unsqueeze(0)
    with torch.no_grad():
        reconstructed_signals=model(signals_tensor)
    label=sample_data[2].tolist()
    label=types[label]
    frequency=sample_data[3].tolist()
    time=np.linspace(0,2,6000)
    source='Wenzhang'
else: #Data is Unseen
    new_sample_index=np.where(new_indices==plotting_index)[0][0]
    signals=new_signals[new_sample_index,:,:]
    with torch.no_grad():
        reconstructed_signals=model(new_signals[new_sample_index,:,:].unsqueeze(0)) 
    label=new_labels[new_sample_index]
    frequency=new_frequencies[new_sample_index]
    time=new_times[new_sample_index,:]*1000
    source='Unseen Data, Airey'

fig, axs = plt.subplots(nrows=3,ncols=1,figsize=(10, 10))
axs[0].set_title(f'Input Signal for: Wave Type: {label}, Frequency: {frequency} Hz, {source}')
axs[0].set_ylabel('Normalised Magnitude')
axs[0].set_xlabel('Time [s]')
axs[0].plot(time,signals[0,:].flatten().tolist(), label='Input Signal')
axs[0].plot(time,reconstructed_signals[0,0,:].flatten().tolist(),linestyle='--', label='Reconstructed Input Signal')
axs[0].legend()
axs[1].set_ylabel('Normalised Magnitude')
axs[1].set_xlabel('Time [s]')
axs[1].set_title('Output Signal')
axs[1].plot(time,signals[1,:].flatten().tolist(), label='Output Signal')
axs[1].plot(time,reconstructed_signals[0,1,:].flatten().tolist(),linestyle='--', label='Reconstructed Output Signal')
axs[1].legend()
axs[2].plot(time,abs(signals[0,:].flatten()-reconstructed_signals[0,0,:].flatten()),label='Input error')
axs[2].plot(time,abs(signals[1,:].flatten()-reconstructed_signals[0,1,:].flatten()),label='Output error')
axs[2].legend()
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Absolute Error')
axs[2].set_title('Original Signal subtract the Reconstructed Signal')
plt.subplots_adjust(hspace=0.5)
st.pyplot(fig)   