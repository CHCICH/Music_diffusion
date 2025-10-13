import torch
import numpy as np
import torch.utils.data as DataLoader
import librosa
import soundfile as sf
import U_net_denoiser as und
import matplotlib.pyplot as plt
import forward_noising as fn

# Suppress warnings that might occur during librosa loading/processing

T = 500
BETAS = torch.linspace(0.0001, 0.02, T)
ALPHAS = 1 - BETAS
A_T_BAR = torch.cumprod(ALPHAS,dim=0)
sqrt_alphas_epsilon = torch.sqrt(1 - A_T_BAR)
sqrt_alphas_x_0 = torch.sqrt(A_T_BAR)


try:
    Y, sr = librosa.load("sample1.mp3", sr=None)
except Exception:
    sr = 22050
    Y = np.random.randn(sr * 2) 

F_stft = np.abs(librosa.stft(Y))
F_db = librosa.amplitude_to_db(F_stft)
F_db = librosa.feature.melspectrogram(S=F_db, sr=sr) 
F_db_tensor = torch.tensor(F_db, dtype=torch.float32)


model_denoiser = und.DenoiserNetwork_Unet()
Loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model_denoiser.parameters(), lr=1e-4) # we tried 1e-4 but the results were not as good
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
batch_size = 2
epochs = 800 # 600 is the first rate of convergence we saw for ADAM optimizer for ADAMW it was 800 but this is only for 1 sample


for epoch in range(epochs):
    print(epoch)

    t = torch.randint(0, T, (batch_size,)) 

    spectrogram_patch = F_db_tensor[:128, :512]
    x_0 = spectrogram_patch.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    x_0 = fn.filter_and_normalize(x_0, spectrogram_patch.min(), spectrogram_patch.max())[0]
    forward_noise = fn.forward_noise_q(x_0, t, sqrt_alphas_x_0, sqrt_alphas_epsilon)
    t = t.float()     
    y_pred = model_denoiser(forward_noise, t)
    loss = Loss(y_pred, x_0)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

denoised_output = None

with torch.inference_mode():
    generated_noise = F_db_tensor[:128, :512]
    generated_noise = generated_noise.unsqueeze(0).unsqueeze(0)
    t = torch.tensor([499])
    min_val = generated_noise.min()
    max_val = generated_noise.max()
    generated_noise = fn.filter_and_normalize(generated_noise, min_val, max_val)[0]
    generated_noise = fn.forward_noise_q(generated_noise, t, sqrt_alphas_x_0, sqrt_alphas_epsilon)
    denoised_output = model_denoiser(generated_noise, t.float())
    denoised_output = fn.reverse_filter_and_normalize(denoised_output, min_val, max_val)


if denoised_output is not None:
    denoised_spectrogram = denoised_output.squeeze().detach().numpy()
    denoised_spectrogram = fn.reverse_filter_and_normalize(denoised_spectrogram, F_db.min(), F_db.max())
    reconstructed_audio = librosa.feature.inverse.mel_to_audio(denoised_spectrogram, sr=sr)
    sf.write("reconstructed_audio_4.wav", reconstructed_audio, sr)
    sf.write("original_audio.wav", librosa.feature.inverse.mel_to_audio(F_db[:128, :512], sr=sr), sr)
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].imshow(F_db[:128, :512], aspect='auto', origin='lower', cmap='viridis')
    axs[0].set_title("Original Spectrogram")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Frequency")
    axs[0].colorbar = plt.colorbar(mappable=axs[0].images[0], ax=axs[0])

    gen_noise = generated_noise.squeeze().detach().numpy()
    axs[1].imshow(gen_noise, aspect='auto', origin='lower', cmap='viridis')
    axs[1].set_title("Generated Noise")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Frequency")
    axs[1].colorbar = plt.colorbar(mappable=axs[1].images[0], ax=axs[1])

    axs[2].imshow(denoised_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axs[2].set_title("Denoised Spectrogram")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Frequency")
    axs[2].colorbar = plt.colorbar(mappable=axs[2].images[0], ax=axs[2])

    plt.tight_layout()
    plt.show()
