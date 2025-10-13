import torch


def get_value_at_index(list_tensor, t, x_shape):
    batch_size = x_shape[0]
    t = t.long()
    out = list_tensor.gather(-1, t)    
    return out.reshape(batch_size, 1, 1, 1)

def forward_noise_q(x_0, t, A1, A2):
    noise = torch.randn_like(x_0)
    sqrt_alphas_x_0_t = get_value_at_index(A1, t, x_0.shape)
    sqrt_alphas_epsilon_t = get_value_at_index(A2, t, x_0.shape)
    x_t = sqrt_alphas_x_0_t * x_0 + sqrt_alphas_epsilon_t * noise
    return x_t

def filter_and_normalize(data, min_val, max_val):
    data = (data - min_val) / (max_val - min_val)
    return data, min_val, max_val

def reverse_filter_and_normalize(data, min_val, max_val):
    data = data * (max_val - min_val) + min_val
    return data
