import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import mido
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pretty_midi
from Model_LSTM import LSTMModel,LSTMDataset
from helper_LSTM import DataToTrain,makeThemNotes,midi_to_pianoroll,train_model
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

piano_roll, times = midi_to_pianoroll('Bach_test.mid',fs=20)
Notes = makeThemNotes(piano_roll)
RawData_ready_to_train = DataToTrain(Notes,64)

Compressed_data_train = RawData_ready_to_train[0]
Compressed_data_target = RawData_ready_to_train[1]

for i in range(len(Compressed_data_train)):
    for j in range(len(Compressed_data_train[i])):
        Compressed_data_train[i][j] = Compressed_data_train[i][j] / 128

Compressed_data_target_int = [int(t) for t in Compressed_data_target]
BATCH_SIZE = 3

DataSet_LSTM_BACH = LSTMDataset(Compressed_data_train, Compressed_data_target_int, device)
DataLoader_LSTM_BACH = DataLoader(DataSet_LSTM_BACH, batch_size=BATCH_SIZE)

print(DataSet_LSTM_BACH.data.shape)
print(DataSet_LSTM_BACH.target.shape)
print(torch.unique(DataSet_LSTM_BACH.target))

complexity = [i for i in range(6,30)]
Loss_graph = []
epochs = 20
hidden_layer = 5
size_of_layer = 6

for i in range(6,30):
    model_used = LSTMModel(1, i, i, 129, device)
    loss_current = train_model(model=model_used,DataLoader=DataLoader_LSTM_BACH, epochs=20)
    Loss_graph.append(loss_current)


device = torch.device('cuda')
model_used.eval()

seed = Compressed_data_train[0] + Compressed_data_train[1]
print(seed)
current = torch.tensor(seed[-64:], dtype=torch.float32).unsqueeze(-1).to(device)

notes = []

with torch.no_grad():
    for _ in range(200):
        logits = model_used(current.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        next_note_class = torch.multinomial(probs, num_samples=1).item()
        notes.append(next_note_class)
        normalized_input = next_note_class / 128.0
        current = torch.cat([current[1:], torch.tensor([[normalized_input]], dtype=torch.float32).to(device)])

midi = pretty_midi.PrettyMIDI()
piano = pretty_midi.Instrument(0)

for i, n in enumerate(notes):
    piano.notes.append(pretty_midi.Note(
        velocity=64,
        pitch=n,
        start=i*0.25,
        end=(i+1)*0.25
    ))

midi.instruments.append(piano)
midi.write('test_fixed_seed_changed_fs_1000.mid')
print(f"Saved {len(notes)} notes to test2.mid")

plt.plot(complexity,Loss_graph)
plt.show()

