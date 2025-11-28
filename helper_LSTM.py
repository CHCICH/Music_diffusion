import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import mido
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pretty_midi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def midi_to_pianoroll(path, fs=100, normalize=False):
    """
    Load a MIDI file and return a piano-roll numpy array and the time axis.
    - path: path to .mid file
    - fs: frames per second (time resolution)
    - normalize: if True, scale velocities to [0,1]
    Returns:
    - piano_roll: shape (T, 128), dtype float32 (T = number of time frames)
    - times: shape (T,), time in seconds for each frame
    """
    pm = pretty_midi.PrettyMIDI(path)
    pr = pm.get_piano_roll(fs=fs)  # shape (128, T), values = velocities (0-127)
    pr = pr.T.astype(np.float32)   # convert to (T, 128)
    if normalize:
        maxv = pr.max() if pr.max() > 0 else 1.0
        pr = pr / maxv
    times = np.linspace(0, pm.get_end_time(), pr.shape[0])
    return pr, times

piano_roll, times = midi_to_pianoroll('Bach_test.mid', fs=100, normalize=True)
# print((piano_roll).shape, times)


def makeThemNotes(Piece):
    NotePerDuation = []
    for piano_rolls in Piece:
        note_played = 128 # in this case we label the note 128 as no notes are being played
        for i in range(len(piano_rolls)):
            note = piano_rolls[i]
            if note > 0:
                note_played = i
        NotePerDuation.append(note_played)
    return NotePerDuation

def DataToTrain(Notes,size_of_the_window):
    # we will perform a sliding window algorithm to solve the problem
    final_data = []
    final_target_data = []
    for i in range(len(Notes) - size_of_the_window):
        window = []
        for j in range(size_of_the_window):
            window.append(Notes[i+j])
        final_data.append(window)
        if i < (len(Notes) - size_of_the_window - 1):
            final_target_data.append(Notes[i+size_of_the_window])
        else:
            final_target_data.append(Notes[len(Notes) - 1])
    return final_data,final_target_data


def train_model(model,DataLoader,epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("---Starting Training Rn---") 
    predictiveGenerativeModel = model
    Loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(predictiveGenerativeModel.parameters(), lr=1e-3)
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        counter = 0
        for inputs, target in DataLoader:
            optimizer.zero_grad()
            outputs = predictiveGenerativeModel(inputs)  
            loss = Loss(outputs, target)                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            counter += 1
            if counter % 100 == 0:
                print(epoch_loss, "at", counter)
        avg_loss = epoch_loss / len(DataLoader)
        total_loss += avg_loss
        print(f"the average loss of the model for epoch {epoch} is {avg_loss}")
    total_loss = total_loss/epoch

    device = torch.device('cuda')