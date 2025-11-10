import torch
import torch.nn as nn

model_Rnn = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
if torch.cuda.is_available():
        print("GPU is available!")
else:
    print("GPU is NOT available. PyTorch will use CPU.")

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
