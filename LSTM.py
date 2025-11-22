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
from helper_LSTM import DataToTrain,makeThemNotes,midi_to_pianoroll, normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

piano_roll, times = midi_to_pianoroll('Bach_test.mid')
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

predictiveGenerativeModel = LSTMModel(1, 6, 5, 129, device)
Loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(predictiveGenerativeModel.parameters(), lr=1e-3)

epochs = 20

print("---Starting Training Rn---")
for epoch in range(epochs):
    epoch_loss = 0
    counter = 0
    for inputs, target in DataLoader_LSTM_BACH:
        optimizer.zero_grad()
        outputs = predictiveGenerativeModel(inputs)  
        loss = Loss(outputs, target)                
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        counter += 1
        if counter % 100 == 0:
            print(epoch_loss, "at", counter)
    avg_loss = epoch_loss / len(DataLoader_LSTM_BACH)
    print(f"the average loss of the model for epoch {epoch} is {avg_loss}")

device = torch.device('cuda')
predictiveGenerativeModel.eval()

seed = [80/128, 123/128, 122/128, 23/128, 122/128, 122/128] * 13
current = torch.tensor(seed[-64:], dtype=torch.float32).unsqueeze(-1).to(device)

notes = []

with torch.no_grad():
    for _ in range(200):
        logits = predictiveGenerativeModel(current.unsqueeze(0))
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
midi.write('test5.mid')
print(f"Saved {len(notes)} notes to test2.mid")





