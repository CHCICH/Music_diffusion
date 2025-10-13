import torch
import pandas as pd
import torch.utils.data as DataLoader
import forward_noising as fn
import librosa as lb



class DataLoaderConditional:
    def __init__(self, file_paths, batch_size, window_size=128, conditions=[]):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.window_size = window_size
        self.conditions = conditions
        self.dataloader = None

    def load_data(self,isMel=False):
        spectrograms = []
        

