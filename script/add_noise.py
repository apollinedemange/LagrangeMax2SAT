import torch


def add_noise(unary_lagrangian, binary_lagrangian, std_unary=0.1, std_binary=0.1):
    ''' Add gaussian noise with a standard deivation std to unary and binary lagrangians. '''
    N = unary_lagrangian.shape[0]
    # Add noise to unary lagrangians
    unary_lagrangian_noisy = unary_lagrangian + torch.rand_like(unary_lagrangian) * std_unary 
    # Add noise to binary lagrangians
    binary_lagrangian_noisy = binary_lagrangian + torch.rand_like(binary_lagrangian) * std_binary
    # Make sure elements on diagonal of binary_lagrangian_noisy are zero
    index = torch.arange(N)
    binary_lagrangian_noisy[index, index, :] = 0
    return unary_lagrangian_noisy, binary_lagrangian_noisy